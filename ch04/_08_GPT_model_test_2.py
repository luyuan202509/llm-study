"""
练习 4.2 初始化更大的 GPT 模型
GPT-2 small
GPT-2 medium 具有 1024 维嵌 入、24 个 Transformer 块和 16 个多头注意力头
GPT-2 large 具有 1280 维嵌 36 个 Transformer 块和 20 个多头注意力头
GPT-2 xl 具有 1600 维嵌 入、48 个 Transformer 块和 25 个多头注意力头
"""
import torch
from torch import nn
import tiktoken

from _07_GPT_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}


def get_config(base_config,model_name = 'gpt-2-small'):
    GPT_CONFIG_124M  = base_config.copy()

    if model_name == 'gpt2-small':
        GPT_CONFIG_124M['emb_dim'] = 768
        GPT_CONFIG_124M['n_layers'] = 12
        GPT_CONFIG_124M['n_heads'] = 12
    elif model_name == 'gpt2-medium':
        GPT_CONFIG_124M['emb_dim'] = 1024
        GPT_CONFIG_124M['n_layers'] = 24
        GPT_CONFIG_124M['n_heads'] = 16
    elif model_name == 'gpt2-large':
        GPT_CONFIG_124M['emb_dim'] = 1280
        GPT_CONFIG_124M['n_layers'] = 36
        GPT_CONFIG_124M['n_heads'] = 20
    elif model_name == 'gpt2-xl':
        GPT_CONFIG_124M['emb_dim'] = 1600
        GPT_CONFIG_124M['n_layers'] = 48
        GPT_CONFIG_124M['n_heads'] = 25
    return GPT_CONFIG_124M

def calculate_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / 1024 / 1024 
    
    print(f"Total size of the model: {total_size_mb:.2f} MB")

if __name__ == "__main__":
    torch.manual_seed(123)


    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    
    #两个文本的词元 ID 序列
    batch.append(tokenizer.encode(text1))
    batch.append(tokenizer.encode(text2))


    for model_abbrev in ('small','medium','large','xl'):
        model_name = f'gpt2-{model_abbrev}'
        cfg = get_config(GPT_CONFIG_124M,model_name = model_name)
        model = GPTModel(cfg)
        print(f"\n\n{model_name}:")
        calculate_size(model)

    

    