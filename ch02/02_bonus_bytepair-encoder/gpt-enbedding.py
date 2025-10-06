import torch
from gpt_dataset import GPTDataset as gd
from RawText import RawText as rt

# 这里模拟词元一个只有4个词元ID的ID列表
#input_ids = torch.tensor([2,3,5,1])
# 获取输入文本
rt = rt()
raw_text = rt.read()


vocab_size = 50257 #词汇表
output_dim = 256 # 输出维度
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)

# 应用到一个词元 ID 上,以获取嵌入向量 这里引用在第三个词元，对应第4行权重
#print(embedding_layer(torch.tensor([1])))

max_length = 4
data_loader = gd.create_dataloader_v1(raw_text,
                                      batch_size=8,
                                      max_length=max_length,
                                      stride = max_length,
                                      shuffle = False
                                      )

data_iter = iter(data_loader)
inputs,targets = next(data_iter)

print("tokenids:\n ",inputs)
print("inputs shape:\n ",inputs.shape)


