import torch.nn as nn 
import torch

class SelfAttention_v1(nn.Module):
    '''
    步骤
    1. 定义
      1.1 词嵌入向量 输入
      1.2 初始化可训练权重矩阵
    2. 得到 query,key,value 向量
    3. 得到分数矩阵,并归一化 得到 attn_weights 权重矩阵
    4. 权重矩阵 乘以 value 向量，得到上下文向量
    '''
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.randn(d_in,d_out),requires_grad=False) # (6,2)
        self.W_key   = nn.Parameter(torch.randn(d_in,d_out),requires_grad=False)
        self.W_value = nn.Parameter(torch.randn(d_in,d_out),requires_grad=False)    

    def forward(self,x):
        keys     = x @ self.W_key   
        queries  = x @ self.W_query
        values   = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1) #(6,6)
        
        context_vec = attn_weights @ values # (6,6)
        return context_vec