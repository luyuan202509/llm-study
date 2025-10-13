
import torch 
import torch.nn as nn
from self_attention_v2 import  SelfAttention_v2
from causal_attention import  CausalAttention
class MultiHeadAttentionWrapper(torch.nn.Module):
    """
    多头注意力机制包装类
    """
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):

        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in,d_out,context_length,dropout,qkv_bias) for _ in range(num_heads)])

    
    def forward(self,x):
        """
        多头注意力机制
        """
        return torch.cat([head(x) for head in self.heads],dim=-1)


    