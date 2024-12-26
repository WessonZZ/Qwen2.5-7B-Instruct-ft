# -------------------------------------------------------------
# ref https://github.com/liuchen6667/qwen2.5_sft_kd
# -------------------------------------------------------------

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

from tokenizer import QADataset

out_dir = 'output-lora'
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

BASE_DIR = 'Qwen2.5-7B-ft'
model_dir = 'Qwen2.5-7B-Instruct'
data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train.json')
val_dir = os.path.join(data_dir, 'val.json')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, torch_dtype=torch.bfloat16)
model.enable_input_require_grads() 
train_data = QADataset(train_dir, Tokenizer, 150, 1024)
val_data = QADataset(val_dir, Tokenizer, 150, 1024)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=16,  # Lora alpha，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir= out_dir,
    eval_strategy='steps',
    eval_steps = 50,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-5,
    lr_scheduler_type= 'cosine',
    # max_steps = 3000,
    warmup_steps=900,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
    save_total_limit=3,
    # load_best_model_at_end=True,
    disable_tqdm=False,
    bf16=True, #fp16=True
    deepspeed=None, #配置deepspeed
    # load_best_model_at_end = True,
    local_rank=os.getenv('LOCAL_RANK', -1),
    ddp_find_unused_parameters=False  # 禁用 DDP 未使用参数检查
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data
)

if __name__=="__main__":
    trainer.train()