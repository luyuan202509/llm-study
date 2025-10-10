'''
因果注意力类测试
'''
import torch 
from self_attention_v1 import  SelfAttention_v1  
from self_attention_v2 import  SelfAttention_v2
from causal_attention import Causal_attention

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2


torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0) # 两个输入,每个输入有 6 个词元,每个词元的嵌入维度为 3
print(batch.shape)
print(batch)
context_length = batch.shape[1]
print(context_length)

print("============================================================================")
ca =  Causal_attention(d_in,d_out,context_length,0.0)
context_vec = ca(batch)

# 最终的上下文向量
print(context_vec.shape)
print(context_vec)


