"""简单自注意力机制"""

import os,sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch 
# 这里其实后续没有用，实际用的是现成的 torch.nn.functional.softmax
from common.my_funcitons import softmax 
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attn_score2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_score2[i] = torch.dot(x_i, query)

#attn_weight_2_tmp = attn_score2 / attn_score2.sum()
#attn_weight_2_tmp = softmax(attn_score2)
# 归一化的注意力权重
attn_weight_2_tmp = torch.softmax(attn_score2,dim=0)



# 上下文向量
context_vec_2 = torch.zeros(query.shape) 
for i,x_i in enumerate(inputs):
    # 词元输入向量*注意力权重
    context_vec_2 += attn_weight_2_tmp[i] * x_i
print(context_vec_2)


