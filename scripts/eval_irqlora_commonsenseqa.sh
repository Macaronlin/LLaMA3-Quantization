tau_range=0.1
tau_n=100
blocksize2=256

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /home/inspur/lin/pretrained_models/llama-3-8b  \
--peft /home/inspur/lin/codes/IR-QLoRA/output/llama-3-8b-irqlora/checkpoint-10000 \
--tau_range ${tau_range} --tau_n ${tau_n} --blocksize ${blocksize2} \
--epochs 0 --output_dir ./log/llama-3-8b-irqlora-${tau_range}-${tau_n}-${blocksize2} \
--wbits 4 \
--tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
