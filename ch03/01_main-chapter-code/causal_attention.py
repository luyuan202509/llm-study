'''
一个简化的因果注意力类
步骤：
1.掩码： 得到分数矩阵后，密对角线上的部分用0 掩码，掩码后 将所有行归一化，得到注意力权重矩阵
2，注意力权重矩阵进行 dropout，
3.生成上下文向量
'''


import torch.nn as nn 
import torch

class Causal_attention(nn.Module):

    def __init__(self,d_in,d_out,context_length,qkv_bias=False,dropout=0.5):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias) # (6,2)
        self.W_key   = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)    
        self.dropout = nn.Dropout(dropout) # 
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length,context_length),diagonal=1)
        )

    def forward(self,x):
        b,num_tokens,d_in = x.shape  # 这里只用到 num_tokens
        keys     = self.W_key(x)
        queries  = self.W_query(x)
        values   = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        # 掩码操作
        attn_scores.masked_fill(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1) #(6,6)
        # 添加dropout
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values # (6,6)
        return context_vec
        
        