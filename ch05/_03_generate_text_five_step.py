import tiktoken  
import torch 
from previous_chapters import generate_text_simple,GPTModel
from RawText import RawText as rt
from gpt_dataset  import GPTDataset as gd
def text_to_token_ids(text,tokenizer):
    '''将文本转换为词元 ID 序列'''
    encoded = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    '''将词元 ID 序列转换为文本'''
    flat = token_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text   

def calc_loss_batch(input_batch,target_batch,model,device):
    """单个批次的损失"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss 

def calc_loss_loader(dataloader,model,device,num_batches=None):
    total_loss = 0.0

    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches,len(dataloader))
    
    for i,(input_batch,target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    context1 = "every effort moves"
    context2 = "I really like"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_context1 = text_to_token_ids(context1,tokenizer)
    encoded_context2 = text_to_token_ids(context2,tokenizer)

    target_text1 = " effort moves you"
    target_text2 = " really like chocolate"
    encoded_target1 = text_to_token_ids(target_text1,tokenizer)
    encoded_target2 = text_to_token_ids(target_text2,tokenizer)



    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]


    model = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(f'probas shape: {probas.shape}') # Shape: (batch_size, num_tokens, vocab_size)
    print(f'probas: {probas}')

    # 获取最大的概率对应的词元 ID
    token_ids =  torch.argmax(probas, dim=-1,keepdim=True)
    print(token_ids)

    print(f"targets batch1:{token_ids_to_text(targets[0],tokenizer)}")
    print(f"Output batch1:{token_ids_to_text(token_ids[0].squeeze(),tokenizer)}")

    print('=='*20)
    # 三维张量的高级索引方式，获取文本1的第0,1,2个词元在目标词元索引处的概率
    text_idx = 0
    target_probas_1 = probas[text_idx,[0,1,2],targets[text_idx]] # 第1维度，第2维度，第3维度
    print(f'Text 1: {target_probas_1}')
    print(f'targets[text_idx]: {targets[text_idx]}')
    
    text_idx = 1
    target_probas_2 = probas[text_idx,[0,1,2],targets[text_idx]]
    print(f'Text 2: {target_probas_2}')
    print(f'targets[text_idx]: {targets[text_idx]}')

    print('=='*20)
    # 对概率分数应用对数
    log_probas = torch.log(torch.cat((target_probas_1,target_probas_2)))
    print(f'log_probas: {log_probas}')

    """
    我们的目标是通过在训练过程中更新模型的权重,使平均对数概率尽可能接近 0。然而,在深度 学习中,通常的做法不是将平均对数概率升至 0,而是将负平均对数概率降至 0。负平均对数概 率就是平均对数概率乘以-1 
    """

    # 计算平均数,目标是通过在训练过程中更新模型的权重,使平均对数概率尽可能接近 0
    avg_log_probas = torch.mean(log_probas)
    print(f'average_log_probas: {avg_log_probas}')
    # 计算平均对数概率的负值
    neg_avg_log_probas = avg_log_probas * -1
    print(f'negative_average_log_probas: {neg_avg_log_probas}')
   

    print('=='*20)
    # 回顾一下 logits 张量和 targets 张量的形状
    print(f'logits shape: {logits.shape}')
    print(f'targets shape: {targets.shape}')
    
    flattened_logits = logits.flatten(0,1)
    print(f'flattened logits shape: {flattened_logits.shape}')


    logits_flat = logits.flatten(0,1)
    targets_flat = targets.flatten()
    print(f'logits_flat: {logits_flat.shape}')
    print(f'targets_flag: {targets_flat.shape}')

    # 计算损失
    loss = torch.nn.functional.cross_entropy(logits_flat,targets_flat)
    print(f'loss: {loss}')

    print('=='*40)
    
    rt = rt()
    text_data = rt.read()
   # print(f'raw_text: {raw_text}')

    total_characters = len(text_data) 
    total_tokens = len(tokenizer.encode(text_data)) 
    print("Characters:", total_characters) 
    print("Tokens:", total_tokens)


    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    print(split_idx)
    train_data = text_data[:split_idx]  # 训练数据
    val_data = text_data[split_idx:]  # 验证数据

   # 创建数据加载器
    print('=='*40)
    torch.manual_seed(123)
    train_loder = gd.create_dataloader_v1(
        train_data,
        batch_size = 2,
        max_length = GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle = True,
        drop_last = True,
        num_workers = 0
    )

    val_loader = gd.create_dataloader_v1(
        val_data,
        batch_size = 2,
        max_length = GPT_CONFIG_124M["context_length"],
        stride = GPT_CONFIG_124M["context_length"],
        shuffle = False,
        drop_last = False,
        num_workers = 0
    )

    print("Train loader size:", len(train_loder))
    for x,y in train_loder:
        print(f"Inputs shape: {x.shape}")
        print(f"Targets shape: {y.shape}")
    
    print("Val loader size:", len(val_loader))
    for x,y in val_loader:
        print(f"Inputs shape: {x.shape}")
        print(f"Targets shape: {y.shape}")


    print(f'计算损失','=='*40)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loder,model,device,num_batches=1)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=1)
    
    print(train_loss)
    print(val_loss)