import json 
import os 
import urllib.request
from torch.utils.data import Dataset
import tiktoken
import torch
from torch.utils.data import DataLoader
from functools import partial

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt

from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)   
    return data

# 格式化输入
def format_input(entry):

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

class InstructionDataset(Dataset):
    def __init__(self, data,tokenizer):
        self.data = data
        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            ) 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
       return self.encoded_texts[idx]


def custom_collate_draft_1(batch,pad_token_id=50256,device="cpu"):
    """自定义数据集的第一个版本:每个批次统一长度"""
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    
    inputs_tensor = torch.stack(inputs_lst)
    return inputs_tensor.to(device)

def custom_collate_draft_2(batch,pad_token_id=50256,device="cpu"):
    """自定义数据集的第2个版本:每个批次统一长度,加上目标序列，以便后续的训练"""
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        inputs_lst.append(inputs)
        targets_lst.append(targets)
    
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def custom_collate_fn(batch,pad_token_id=50256,ignore_index=-100,allowed_max_length=None,device="cpu"):
    batch_max_length = max(len(item) +1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = (
            new_item + [pad_token_id] * 
           (batch_max_length - len(new_item)))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor



def main1():
    """下载数据集"""
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(custom_collate_fn,
                                                           device = device,
                                                           allowed_max_length=1024)

    ## 数据集划分

    train_data = data[:int(len(data) * 0.8)]
    val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]


    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = InstructionDataset(train_data,tokenizer)
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers)

    val_dataset = InstructionDataset(val_data,tokenizer)
    val_loader = DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers)

    test_dataset = InstructionDataset(test_data,tokenizer)
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers)
    
    print("输入数据dataloader: \n ")
    for inputs,targets in train_loader:
        print(inputs[0])
        print(targets[0])
        break 
    
    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

def main2():
    """下载数据集"""
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(custom_collate_fn,
                                                           device = device,
                                                           allowed_max_length=1024)

    ## 数据集划分

    train_data = data[:int(len(data) * 0.8)]
    val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]


    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = InstructionDataset(train_data,tokenizer)
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers)

    val_dataset = InstructionDataset(val_data,tokenizer)
    val_loader = DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers)

    test_dataset = InstructionDataset(test_data,tokenizer)
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   collate_fn=customized_collate_fn,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers)
    
    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    



if __name__ == "__main__":
    main1()
