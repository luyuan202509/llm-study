"""简单自注意力机制"""

import os,sys 
import torch 

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

attn_scores = torch.empty((6,6))
"""
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
       attn_scores[i,j] = torch.dot(x_i, x_j)
"""

# 矩阵乘法 更简单，更高效
attn_scores = inputs @ inputs.T

# 归一化
attnweight = torch.softmax(attn_scores,dim=1)

# 上下文向量
all_context_vec =attnweight @ inputs
print(all_context_vec)
