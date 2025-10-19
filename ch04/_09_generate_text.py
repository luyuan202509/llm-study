import torch 
import tiktoken


from _07_GPT_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,            # 嵌 入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # 丢弃率
    "qvb_bias": False,       # 查询 键 值 偏置
}


def generate_text_simple(model,idx,max_new_tokens,context_size):
    '''
    简单生成文本
    Args:
        model: 模型
        idx: 输入的词元 ID 序列
        max_new_tokens: 最大新词元数量
        context_size: 上下文大小
    Returns:
        idx: 生成的词元 ID 序列
    '''
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probas = torch.softmax(logits,dim= -1)
        idx_next = torch.argmax(probas,dim= -1,keepdim=True)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx



if __name__ == "__main__":
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    start_context = "Hello, I am"
    incoded = tokenizer.encode(start_context)
    incoded_tensor = torch.tensor(incoded).unsqueeze(0)

    cfg = GPT_CONFIG_124M
    gpt_model = GPTModel(cfg)
    
    gpt_model.eval()
    out  = generate_text_simple(model = gpt_model,
                                idx = incoded_tensor,
                                max_new_tokens= 6,
                                context_size=GPT_CONFIG_124M["context_length"])

    # 得到生成的词元 ID 序列                           
    print(f'Output: {out}')
    print(f'output length:{len(out[0])}')

    # 将词元 ID 序列转换为文本  
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f'Decoded text: {decoded_text}')  # Hello, I am Featureiman Byeswick Exit In 
