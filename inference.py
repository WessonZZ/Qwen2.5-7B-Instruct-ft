# -------------------------------------------------------------
# reference with fine-tuned model
# -------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import os

def predict(text, model, tokenizer, device):
    messages = [
                {"role": "system",
                "content": "你是一个智慧问答助手，能够利用所学知识正确地回答用户的问题。"},
                {"role": "user", "content": text}
            ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        temperature = 0.9,
        max_new_tokens=1024)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main(ft = True):
    model_dir = 'Qwen2.5-7B-Instruct'
    ft_model_dir = 'output-lora'
    Tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if ft: #finetuned model
        print("---------fine-tuned model-----------")
        model = AutoModelForCausalLM.from_pretrained(ft_model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    else: #un-finetuned model
        print("---------un-fine-tuned model-----------")
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Text = [
        "你是谁？",
        "芝加哥的人会觉得UIUC超土吗？",
        "什么是模态分析？",
        "如何评价打工诗人许立志及其诗？",
        "如何看待 iPhone 12 mini？"
    ]
    
    for text in Text:
        response = predict(text, model, Tokenizer, device)
        print("----------------------------------")
        print(f"Input: {text}\nResponse: {response}")
        
if __name__=="__main__":
    ft = True
    main(ft)
    