# Qwen2.5-7B-Instruct-ft
The repository contains some basic scripts and datasets to fine-tune Qwen2.5-7B-Instruct model.

Follow below instructions to fine-tune your own LLM.
```
1. python data_preprocess.py
2. python add_data.py
3. python finetune_lora.py / torchrun --nproc_per_node=2 --master_port=12345 finetune_lora.py
4. python inference.py
```
