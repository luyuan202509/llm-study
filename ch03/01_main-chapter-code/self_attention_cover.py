''' 
因果注意力的掩码实现 
步骤：
1 注意力分数，未归一化，->(应用softmax) 
2 将对角线上的部分用0 掩码，
3 将所有行归一化，得到注意力权重矩阵
'''

import torch 
from self_attention_v1 import  SelfAttention_v1  
from self_attention_v2 import  SelfAttention_v2

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
sa_v2=  SelfAttention_v2(d_in,d_out)

querys = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = querys @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1)
print(attn_weights)


'''掩码操作'''
context_length = attn_scores.shape[0]
masked_simple = torch.tril(torch.ones(context_length,context_length))

row_sums = masked_simple.sum(dim=1,keepdim=True)
mask_simple_norm = masked_simple / row_sums
print("普通掩码操作",mask_simple_norm)

'''更高效的掩码操作'''
mask = torch.tril(torch.ones(context_length,context_length),diagonal=1)
masked = attn_scores.masked_fill(mask==0,-torch.inf)
print("更高效的掩码操作",masked)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5,dim=-1)
print("更高效的掩码操作",attn_weights)


print("============================================================================")
'''最后生成上下文向量 '''
values = sa_v2.W_value(inputs)
all_context_vec = attn_weights @ values 
print("掩码后的上下文向量",all_context_vec)







