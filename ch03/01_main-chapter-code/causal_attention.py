'''
一个简化的因果注意力类
'''


import torch.nn as nn 
import torch

class Causal_attention(nn.Module):

    def __init__(self,d_in,d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias) # (6,2)
        self.W_key   = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)    

    def forward(self,x):
        keys     = self.W_key(x)
        queries  = self.W_query(x)
        values   = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1) #(6,6)
        
        context_vec = attn_weights @ values # (6,6)
        return context_vec