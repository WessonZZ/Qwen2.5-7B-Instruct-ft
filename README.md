# Qwen2.5-7B-Instruct-ft
The repository contains some basic scripts and datasets to fine-tune Qwen2.5-7B-Instruct model.

- Dataset: My data sets are from https://github.com/esbatmop/MNBVC, maybe you can use them.
**Attention**: you should follow the proper data format, like:
```json
{"input": " ",
 "output": " "} 
```

- Ft method: lora.
- GPU: V100 * 1 or more.
- Framework: peft/transformers

Follow below instructions to fine-tune your own LLM.

Before running these scripts, you should create a new folder "dataset" to contains your dataset.


0. install necessary package
```bash
pip install -r requirements.txt
```
1. pre-process data set
```bash
python data_preprocess.py
python add_data.py
```
2. fine-tune LLM
```bash
python finetune_lora.py   #single GPU
torchrun --nproc_per_node=2 --master_port=12345 finetune_lora.py   #2 GPUs
python finetune_lora_ds.py  #deepspeed
```
3. inference
```bash
python inference.py   #adjust your model dir if necessary
```
