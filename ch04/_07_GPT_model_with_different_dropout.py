"""
GPT 模型
包括：词元嵌入、位置嵌入、随机丢弃、Transformer 块、层归一化、线性输出层

Exercise 4.3: Using separate dropout parameters
在本章开头,我们在 GPT_CONFIG_124M 字典中定义了一个全局的 drop_rate 设置来控  制 GPTModel 架构中各个位置的 dropout 率。
请修改代码,为模型架构中的不同 dropout 层指 定不同的 dropout 值。
(提示:模型中有 3 个不同的 dropout 层:嵌入层、快捷连接层和多头注意力模块。

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_emb": 0.1,        # NEW: dropout for embedding layers
    "drop_rate_attn": 0.1,       # NEW: dropout for multi-head attention  
    "drop_rate_shortcut": 0.1,   # NEW: dropout for shortcut connections  
    "qkv_bias": False
}

"""

#====================================================GPT 模型===================================================
#=== GPT 模型实现
#====================================================GPT 模型===================================================


class GPTModel(nn.Module):
    """ GPT 模型：包括词元嵌入、位置嵌入、随机丢弃、Transformer 块、层归一化、线性输出层 """
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias = False)
        
    def forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len,device = in_idx.device))
       
       # 词元嵌入 + 位置嵌入
        x = tok_emb + pos_emb

        x = self.drop_emb(x) 
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
        
      
#====================================================GPT 模型中重要模块====================================================
#=== 此处主演是把前面的模块重新移到这里，不想分散在各处各个类里
#====================================================GPT 模型中重要模块====================================================

class MultiHeadAttention(torch.nn.Module):
    """
    多头注意力机制包装类
    """
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0,'dout must be divisible by num_heads'
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key   = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)  
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)

        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)

        # 掩码
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # 归一化
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5,dim=-1)
        # droupout 
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(
            b,num_tokens,self.d_out
        )

        context_vec = self.out_proj(context_vec)

        return context_vec

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


class TransformerBlock(nn.Module):
    """ Transformer 块：包括多头注意力、前馈神经网络、层归一化、快捷连接 """
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['n_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qvb_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut  = nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
       
        # 添加 快捷连接
        shortcut = x 
        # 层归一化
        x = self.norm1(x)
        # 多头注意力
        x = self.att(x)
        # 添加 dropout
        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        return x

class LayerNorm(nn.Module):
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