'''
前馈神经网络的介绍：
包括：线性层-> 激活层-> 线性层 
'''

print("====前馈神经网络 GPT模块中重要模块===========================================================")

import torch  
from _01_dummyGPT_model import  FeedForward

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}


ffn = FeedForward(GPT_CONFIG_124M)

x = torch.rand(2,3,768)
out = ffn(x)
print(x.shape)



