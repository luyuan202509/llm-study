import tiktoken 
import torch 
from _01_dummyGPT_model import DummyGPTModel



GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}




tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

#两个文本的词元 ID 序列
batch.append(tokenizer.encode(text1))
batch.append(tokenizer.encode(text2))
print(batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(torch.tensor(batch))
print(f'shape of model output: {logits.shape}')
print(f'logits: {logits}')


