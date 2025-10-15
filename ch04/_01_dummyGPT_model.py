import torch
from torch import nn

'''
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "d_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}
'''

class DummyGPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # 词元嵌入
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        # 位置嵌入
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        # 随机丢弃
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        # 一些列 Transformer 块
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        # 最终层归一化
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        # 线性输出层
        self.out_head = nn.Linear(cfg['emb_dim'],cfg["vocab_size"],bias = False)

    def forward(self,in_index):
        batch_size,seq_len = in_index.shape
        print(f"输入形状：{in_index.shape}")
        # 生产词嵌入向量
        tok_embeds = self.tok_emb(in_index)
        # 生成位置嵌入向量
        pos_embeds = self.pos_emb(torch.arange(seq_len,device =in_index.device))
        # 生成嵌入向量
        x = tok_embeds + pos_embeds
        # 随机丢弃
        x = self.drop_emb(x)
        # 其他层，ransformer块
        x = self.trf_blocks(x)
        # 层归一化
        x = self.final_norm(x)
        # 输出层
        logits = self.out_head(x)
        return logits





class DummyTransformerBlock(nn.Module):
    '''
    transformer 块是 大语言模型的重要关键组成部分
    '''
    def __init__(self,cfg):
        super().__init__()
    
    def forward(self,x):
        return x

class DummyLayerNorm(nn.Module):
    '''层归一化：'''
    def __init__(self,emb_dim):
        super().__init__()
        self.eps = 1e-5 # 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True, unbiased=False)

        nor_x = (x - mean) / torch.sqrt(var + self.eps)  # 减去均值，结果除以方差的平方根

        return self.scale * nor_x + self.shift

class GELU(nn.Module):
    """
    GELU激活函数  类似与ReLU sigmoid
    """
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 0.5 * x * (1+ torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x+ 0.44715* torch.pow(x,3))
        ))

class FeedForward(nn.Module):
    '''前馈神经网络 '''
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'],4*cfg['emb_dim']),
            GELU(),
            nn.Linear(4*cfg['emb_dim'],cfg['emb_dim'])
        )
    def forward(self,x):
        return self.layers(x)
