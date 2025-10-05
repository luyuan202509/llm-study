import torch

# 这里模拟词元一个只有4个词元ID的ID列表
input_ids = torch.tensor([2,3,5,1])

vocab_size = 6 #小型词汇表
output_dim = 4 # 输出维度
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)

# 应用到一个词元 ID 上,以获取嵌入向量 这里引用在第三个词元，对应第4行权重
print(embedding_layer(torch.tensor([3])))


# 应用到一个词元 ID 上,以获取嵌入向量 这里引用在第三个词元，对应第4行权重
print(embedding_layer(torch.tensor([3])))

# 应用到所有词元 ID 
print(embedding_layer(input_ids))