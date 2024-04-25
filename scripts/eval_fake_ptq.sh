# for fake quantization here: AWQ, QuIP, BiLLM, PB-LLM, DB-LLM 
model_path='LLMQ/LLaMA-3-8B-BiLLM-1.1bit-fake'
python main.py --model ${model_path} --epochs 0 --output_dir ./log/--tasks 'hellaswag,piqa,winogrande,arc_easy,arc_challenge' --wbits 16 --abits 16 --eval_ppl --multigpu
