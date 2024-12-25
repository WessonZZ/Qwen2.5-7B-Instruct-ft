# -------------------------------------------------------------
# construct dataset
# -------------------------------------------------------------
import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt


class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length

        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                datas = json.load(f)
                for data in datas:
                    text = data["input"]
                    label = data["output"]
                    self.data.append({
                        "input": text,
                        "output": label
                    })
        print("data loaded , size:", len(self.data))


    @classmethod
    def tokenize(cls, input: str, output: str, Tokenizer, input_max_length = 200, output_max_length = 2896):
        messages = [
                {"role": "system",
                "content": "你是一个智慧问答助手，能够利用所学知识正确地回答用户的问题。"},
                {"role": "user", "content": input}
            ]
        prompt = Tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = Tokenizer(prompt, add_special_tokens=False, max_length=input_max_length,
                                        padding="max_length", pad_to_max_length=False, truncation=True)
        response = Tokenizer(output, add_special_tokens=False, max_length=output_max_length,
                                    padding="max_length", pad_to_max_length=False, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [Tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [Tokenizer.pad_token_id]
        input_length = sum(instruction["attention_mask"])
        output_length = sum(response["attention_mask"])
        
        return input_ids, attention_mask, labels, input_length, output_length

    def __getitem__(self, index):
        item_data = self.data[index]

        input_ids, attention_mask, labels, _, _ = self.__class__.tokenize(item_data["input"], item_data["output"], self.tokenizer, self.max_source_length, self.max_target_length)

        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)



if __name__=="__main__":  
    Tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=False, trust_remote_code=True)
    data_path = "dataset/train.json"
    # input_l, output_l = [0] * 10000, [0] * 10000
    # # input_l, output_l = [0] * 1589370, [0] * 1589370
    # with open(data_path, 'r', encoding='utf-8') as f:
    #             datas = json.load(f)
    #             i = 0
    #             for data in datas:
    #                 input_ids, attention_mask, labels, input_length, output_length = tokenizer(data['input'], data['output'], Tokenizer, 12800, 12800)
    #                 input_l[i] = input_length
    #                 output_l[i] = output_length
    #                 i += 1
    #                 if i % 5000 == 0:
    #                     print(f"{i}")
    # fig, (ax1, ax2) = plt.subplots(1, 2, dpi = 400, sharey= True)
    # ax1.hist(input_l, edgecolor='black', alpha=0.7)
    # ax2.hist(output_l, edgecolor='black', alpha=0.7)
    # ax1.set_xlabel('input length')
    # ax1.set_ylabel('Frequency')
    # ax2.set_xlabel('output length')
    # # ax2.set_ylabel('Frequency')
    # plt.tight_layout()
    # plt.savefig('dataset/text length distribution-val.jpg')
    # # plt.savefig('dataset/text length distribution-train.jpg')
    # plt.show()
    
    QAdata = QADataset(data_path, Tokenizer, 200, 2896)
                


