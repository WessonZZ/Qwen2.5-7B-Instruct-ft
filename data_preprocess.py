# -------------------------------------------------------------
# pre-process our original data, our data are from https://github.com/esbatmop/MNBVC. We appreaciate these guys' hrad work and selfless dedication.
# The full data set is composed by three parts: 1.zhihu qa; 2.wikihow qa; 3.chatgpt qa; 4.mfa qa.
# -------------------------------------------------------------


import json
import os
import random

def _extract_keys(file_data: dict):
    """extract questions and answers from original data

    Args:
        file_data (dict): data
    """
    return (file_data["问"], file_data["答"])


def extract_data(file_path: list):
    """ extract and construct the full data set

    Args:
        file_path (list): original data set's file path
    """
    total_num = 0
    full_data = []
    input_length, output_length = 0, 0
    output_l = 0
    for file in file_path:
        file_name = file.split("/")[1].split(".")[0]
        print(f"---------Starting processing {file_name}--------")
        with open(file, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                data = json.loads(line)
                final_data = {"instruction":"", "input":"", "output":""}
                final_data["input"], final_data["output"] = _extract_keys(data)
                input_length, output_length = max(input_length, len(final_data["input"].split(" "))), max(output_length, len(final_data["output"].split(" ")))
                if len(final_data["output"].split(" ")) > 1024:
                    output_l += 1
                full_data.append(final_data)
                        
                i += 1
        total_num += i
        print(f"---------{file_name} has {i} rows data--------")
    print(f"===================Total {total_num} rows data===============")
    print(f"Max length of input: {input_length}, max length of output: {output_length}")
    print(f"There are {output_l} text longer than 1024")
    random.shuffle(full_data)
    return full_data

if __name__ == "__main__":
    dir_list = ["dataset/"+f for f in os.listdir('dataset') if f.endswith('jsonl')]
    full_data = extract_data(dir_list)

    with open('dataset/train.json', 'w', encoding='utf-8') as train_f,\
        open('dataset/val.json', 'w', encoding='utf-8') as val_f:
        json.dump(full_data[:-10000], train_f, ensure_ascii=False)
        json.dump(full_data[-500:], val_f, ensure_ascii=False)
        