import os,sys 
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE / 'ch03' / '01_main-chapter-code'))

from multi_head_attention import MultiHeadAttention
from _01_dummyGPT_model import FeedForward,DummyLayerNorm as LayerNorm

import torch
from torch import nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qvb_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut  = nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
       
        # 添加 快捷连接
        shortcut = x 
        # 层归一化
        x = self.norm1(x)
        # 多头注意力
        x = self.att(x)
        # 添加 dropout
        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x


if __name__ == "__main__":
    
    torch.manual_seed(123)
    x = torch.rand(2,4,768)
    cfg = GPT_CONFIG_124M
    transformer_block = TransformerBlock(cfg)
    output  = transformer_block(x)
    print(f"输出形状：{output.shape}")
    print(f"输出：{output}")

