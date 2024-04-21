from tqdm import tqdm
import peft
import torch
import operator
import numpy as np
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer
from functools import reduce  # Required in Python 3
import bitsandbytes.functional as bnb_F
from torch import Tensor
from scipy.stats import norm
from bitsandbytes.functional import create_fp8_map, create_dynamic_map

cache_folder_path = ''
module_num = 0
sigma = 1 / norm.ppf(torch.linspace(0.9677083, 0.5, 9)[:-1]).tolist()[0]

def replace_to_qlora_model(model, model_fp, blocksize2=256, tau_range=0.1, tau_n=100):
    model.model = _replace_with_ours_lora_4bit_linear(model.model, model_fp=model_fp, blocksize2=blocksize2, tau_range=tau_range, tau_n=tau_n)
    return model

def prod(iterable):
    return reduce(operator.mul, iterable, 1)
    
normal_map_fp8 = create_dynamic_map()
def quantize_tensor(X, L, idx=False):
    L = L.to(X.device)
    X_shape = X.shape
    X_expanded = X.reshape(-1, 1)
    L_reshaped = L.reshape(1, -1)
    abs_diff = torch.abs(X_expanded - L_reshaped)
    min_index = torch.argmin(abs_diff, dim=-1)
    min_index = torch.tensor(min_index, dtype=torch.uint8).to(L.device).reshape(X_shape)
    return min_index

def dequantize_tensor(X, L):
    L = L.to(X.device)
    return torch.index_select(L, dim=0, index=torch.as_tensor(X, dtype=torch.int32).reshape(-1)).reshape(X.shape)

@torch.no_grad()
def nf4_quant(weight, weight_shape, tau, compress_statistics, quant_type, device):
    weight = weight.reshape(-1, 256, 64).to(device)
    tau = tau.reshape(-1, 256, 1).to(device)
    _weight = (weight - tau).reshape(weight_shape)
    nf4_weight = bnb.nn.Params4bit(_weight, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type).cuda(0)
    tau2 = tau.abs().max(dim=1, keepdim=True)[0]
    tau1 = quantize_tensor(tau / tau2, normal_map_fp8)
    return nf4_weight, tau1.reshape(-1, 256), tau2.reshape(-1, 1)

@torch.no_grad()
def evaluate_entropy(weight_int8, blocksize):
    device = weight_int8.device
    _weight_int8 = weight_int8.reshape(-1, 1)
    weight_nf4 = torch.cat((_weight_int8//16, _weight_int8%16), 1).reshape(1, -1, blocksize)
    weight_nf4_repeat = weight_nf4.repeat(16, 1, 1).to(device)
    values = torch.tensor(range(16)).reshape(16, 1, 1).to(device)
    freqs = (weight_nf4_repeat==values).sum(dim=-1, keepdim=True) / blocksize
    entropy = -freqs * torch.log2(freqs)
    entropy = torch.where(torch.isnan(entropy), 0, entropy)
    entropy = entropy.sum(dim=0)
    return entropy

@torch.no_grad()
def search(fp_weight: Tensor, fp_weight_shape, compress_statistics, quant_type, device, tau_range=0.1, tau_n=51, blocksize=64, blocksize2=256):
    fp_weight = fp_weight.reshape(-1, blocksize2, blocksize).to(device)
    tau0 = fp_weight.median(2, keepdim=True)[0] # [-1, 256, 1]
    absmax = (fp_weight - tau0).abs().max(2, keepdim=True)[0]
    
    entropy_max, factor_best = None, None
    for factor in tqdm(np.linspace(-tau_range*sigma, tau_range*sigma, tau_n*2+1)):
        tau = factor * absmax + tau0
        nf4_weight, _, _ = nf4_quant(fp_weight, fp_weight_shape, tau, compress_statistics, quant_type, device)
        entropy = evaluate_entropy(nf4_weight, blocksize)
        
        if entropy_max is None:
            entropy_max = entropy
            factor_best = torch.full_like(entropy, factor)
        else:
            factor_best = torch.where(entropy > entropy_max, factor, factor_best)
            entropy_max = torch.max(entropy_max, entropy)
    
    tau = factor_best.reshape(-1, 256, 1) * absmax + tau0
    nf4_weight, tau1, tau2 = nf4_quant(fp_weight, fp_weight_shape, tau, compress_statistics, quant_type, device)
    return nf4_weight, tau1, tau2

class IRQLoraLinear4bit(bnb.nn.Linear4bit, LoraLayer):
    def __init__(
        self, old_model, model_fp=None, blocksize2=256, tau_range=0.1, tau_n=51
    ):
        for key, value in old_model.__dict__.items():
            setattr(self, key, value)

        fp_weight = model_fp.weight.data.contiguous().to('cpu')
        fp_weight_shape = fp_weight.shape
        
        compress_statistics, quant_type, device = self.base_layer.weight.compress_statistics, self.base_layer.weight.quant_type, self.base_layer.weight.device
        del self.base_layer.weight, model_fp
        torch.cuda.empty_cache()

        self.base_layer.weight, self.base_layer.tau_quant, self.base_layer.tau_absmax = search(
            fp_weight=fp_weight,
            fp_weight_shape=fp_weight_shape,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            device=device,
            tau_range=tau_range, tau_n=tau_n,
            blocksize2=blocksize2
        )
        self.base_layer.tau_quant = self.base_layer.tau_quant.to(device)
        self.base_layer.tau_absmax = self.base_layer.tau_absmax.to(device)
        
        del fp_weight
        torch.cuda.empty_cache()
        
        self.lora_default_A_scale = torch.nn.Parameter(torch.zeros([1], dtype=self.lora_A.default.weight.dtype).to(self.base_layer.weight.device), requires_grad=True)
        self.lora_default_B_scale = torch.nn.Parameter(torch.zeros([1], dtype=self.lora_A.default.weight.dtype).to(self.base_layer.weight.device), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        
        if self.base_layer.bias is not None and self.base_layer.bias.dtype != x.dtype:
            self.base_layer.bias.data = self.base_layer.bias.data.to(x.dtype)

        if getattr(self.base_layer.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        inp_dtype = x.dtype
        if self.base_layer.compute_dtype is not None:
            x = x.to(self.base_layer.compute_dtype)

        bias = None if self.base_layer.bias is None else self.base_layer.bias.to(self.base_layer.compute_dtype)
        
        with torch.no_grad():
            fp_B = bnb_F.dequantize_fp4(self.base_layer.weight, self.base_layer.weight.quant_state).to(x.dtype)
            tau = (dequantize_tensor(self.base_layer.tau_quant, normal_map_fp8).reshape(-1, 256, 1) * self.base_layer.tau_absmax.reshape(-1, 1, 1)).to(fp_B.device)
            blocksize = torch.prod(torch.tensor(fp_B.shape)) / torch.prod(torch.tensor(tau.shape))
            fp_B = (fp_B.reshape(-1, blocksize.int().item()) + tau.reshape(-1, 1)).reshape(fp_B.shape).to(x.dtype)
        out = torch.nn.functional.linear(x, fp_B, bias)

        out = out.to(inp_dtype)
        result = out
        
        if self.disable_adapters or self.active_adapter[0] not in self.lora_A.keys():
            return result
        elif self.r[self.active_adapter[0]] > 0:
            result = result.clone()
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype
                x = x.to(self.lora_A[self.active_adapter[0]].weight.dtype)
                x = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x)) + self.lora_default_A_scale * x.reshape([_ for _ in x.shape[:-1]] + [self.lora_A[self.active_adapter[0]].out_features] + [-1]).mean(dim=-1)
                x = (self.lora_B[self.active_adapter[0]](x).reshape([_ for _ in x.shape] + [-1]) + self.lora_default_B_scale * x.unsqueeze(-1)).reshape([_ for _ in x.shape[:-1]] + [-1])
                output = x.to(expected_dtype) * self.scaling[self.active_adapter[0]]
            else:
                x = self.lora_A[self.active_adapter[0]](self.lora_dropout[self.active_adapter[0]](x)) + self.lora_default_A_scale * x.reshape([_ for _ in x.shape[:-1]] + [self.lora_A[self.active_adapter[0]].out_features] + [-1]).mean(dim=-1)
                x = (self.lora_B[self.active_adapter[0]](x).reshape([_ for _ in x.shape] + [-1]) + self.lora_default_B_scale * x.unsqueeze(-1)).reshape([_ for _ in x.shape[:-1]] + [-1])
                output = x * self.scaling[self.active_adapter[0]]
            result += output

        return result

def _replace_with_ours_lora_4bit_linear(
    model, current_key_name=None, model_fp=None, blocksize2=256, tau_range=0.5, tau_n=51
):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, peft.tuners.lora.Linear4bit):
            model._modules[name] = IRQLoraLinear4bit(model._modules[name], model_fp=model_fp._modules[name], blocksize2=blocksize2, tau_range=tau_range, tau_n=tau_n)
        
        if len(list(module.children())) > 0:
            if name in model_fp._modules:
                _ = _replace_with_ours_lora_4bit_linear(
                    module,
                    current_key_name, model_fp._modules[name], blocksize2, tau_range, tau_n
                )
            else:
                _ = _replace_with_ours_lora_4bit_linear(
                    module,
                    current_key_name, None, blocksize2, tau_range, tau_n
                )
        current_key_name.pop(-1)
    return model
