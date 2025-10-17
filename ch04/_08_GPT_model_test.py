"""
GPT 模型
包括：词元嵌入、位置嵌入、随机丢弃、Transformer 块、层归一化、线性输出层
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

if __name__ == "__main__":
    torch.manual_seed(123)


    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    
    #两个文本的词元 ID 序列
    batch.append(tokenizer.encode(text1))
    batch.append(tokenizer.encode(text2))

    cfg = GPT_CONFIG_124M
    gpt_model = GPTModel(cfg)
    
    out = gpt_model(torch.tensor(batch,dtype=torch.long))
    print(f"input batch \n {batch}")
    print(f'\n output shape: {out.shape}')
    print(out)

    # 统计模型参数张量的总参数量
    print(f'统计模型参数张量的总参数量\n')
    total_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")    


    print({gpt_model.tok_emb.weight.shape})
    print({gpt_model.out_head.weight.shape})
    