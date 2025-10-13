from sympy.printing.repr import ReprPrinter
import torch
from torch.onnx.symbolic_opset9 import dim


# 词嵌入向量 
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 输入向量
query = inputs[1]
# 第二个向量注意力权重
attn_score_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
  attn_score_2[i] = torch.dot(x_i,query)

print(attn_score_2)

# 归一化的注意力权重
attn_weights_2 = torch.softmax(attn_score_2,dim=0)

# 上下文向量： Z
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
  context_vec_2 += attn_weights_2[i] * x_i 




attn_scores = torch.empty(6,6)
attn_scores = inputs @ inputs.T
""" 
for i,x_i in enumerate(inputs):
  for j,x_j in enumerate(inputs):
    attn_scores[i,j] = torch.dot(x_i,x_j)
"""

attn_weights = torch.softmax(attn_scores,dim=-1)
#attn_weights = torch.softmax(attn_scores, dim=-1)

# 所有 词嵌入向量 的所有上下文向量 
all_context_vec = attn_weights @ inputs
print(all_context_vec)

# 至此，所有简单自注意力机制的代码的讲解到此结束，接下来介绍的是，如何实现可训练的自注意力机制。   

