'''
可训练的自注意力机制 版本2  
    步骤
    1. 定义
      1.1 词嵌入向量 输入
      1.2 初始化可训练权重矩阵
    2. 得到 query,key,value 向量
    3. 得到分数矩阵,并归一化 得到 attn_weights 权重矩阵
    4. 权重矩阵 乘以 value 向量，得到上下文向量

修改：
*可训练权重W_query,W_key,W_value 的初始化 ，使用 nn.Linear 实现,nn.Linear 的一个重要优势是它提供了优化的权重初始化方案,从而有助 于模型训练的稳定性和有效性

'''
import torch.nn as nn 
import torch

class SelfAttention_v2(nn.Module):

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