"""多头注意力机制测试"""

from ast import Mult
import torch 
from multi_head_attention_wrapper import MultiHeadAttentionWrapper
from multi_head_attention import MultiHeadAttention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0) # 两个输入,每个输入有 6 个词元,每个词元的嵌入维度为 3

context_length = batch.shape[1] # 词元数量

d_in,d_out = 3,2
mha = MultiHeadAttentionWrapper(d_in,d_out,context_length,0.0,num_heads=2)

context_vecs = mha(batch)
# 最终的上下文向量
print(context_vecs.shape)
print(context_vecs)



