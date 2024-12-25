# Qwen2.5-7B-Instruct-ft
The repository contains some basic scripts and datasets to fine-tune Qwen2.5-7B-Instruct model.

Follow below instructions to fine-tune your own LLM.
1. pre-process data set
```
python data_preprocess.py
python add_data.py
```
2. fine-tune LLM
```
python finetune_lora.py #single GPU
torchrun --nproc_per_node=2 --master_port=12345 finetune_lora.py #2 GPU
```
3. inference
```
python inference.py #adjust your model dir if necessary
```
