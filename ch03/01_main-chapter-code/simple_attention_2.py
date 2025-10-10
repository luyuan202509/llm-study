import torch


# 词嵌入向量 
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1] 
d_in = inputs.shape[1] #  
dout = 2 

########################################
#############   权重矩阵    #############
########################################
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,dout),requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in,dout),requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in,dout),requires_grad=False)


print("========================query向量=========================================")

key_2   = x_2 @ W_key
query_2 = x_2 @ W_query # (1,2)
value_2 = x_2 @ W_value

keys = inputs @ W_query
values = inputs @ W_value

print(query_2)



print("=======================score 分数向量=======================================")
########################################
#############   注意力分数    ############
########################################
keys_2 = keys[1]
# query 2 的注意力分数
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

########################################
#####   注意力分数转化为注意力权重  ########
########################################
print("==========================分数向量转转为【权重向量】并归一化=====================")

# 得到 归一化的注意力权重 
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5,dim=-1)
print(attn_weights_2)


print("==========================分上下文向量======================================")

# 最终 query_2 的上下文向量
context_vec_2 = attn_weights_2 @ values
print("最终query_2 的上下文向量: ",context_vec_2)
