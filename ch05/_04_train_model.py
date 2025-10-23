import torch    
from previous_chapters import generate_text_simple,GPTModel
from gpt_dataset import GPTDataset as gd
from RawText import RawText as rt
import tiktoken


def train_model_simple(model,train_loader,val_loader,
                       optimizer,device,num_epochs,
                       eval_freq,eval_iter,start_context,tokenizer):
    train_losses,val_losses,track_tokens_seen  = [],[],[]
    tokens_seen,global_step = 0,-1

    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)
                print(f'Ep {epoch+1 } (step{global_step:06d}): '
                      f'Train loss {train_loss: .3f}, '
                      f'Val loss   {val_loss: .3f}')
    
        generate_and_print_sample(model,tokenizer,device,start_context)
    
    return train_losses,val_losses,track_tokens_seen




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


def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    """评估模型：每次模型更新后打印训练集和验 证集的损失,以便我们可以评估训练是否改善了模型性能 """
    model.eval() # 禁用梯度跟踪
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader,model,device,num_batches=eval_iter)
    model.train()
    return train_loss,val_loss


def generate_and_print_sample(model,tokenizer,device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model = model,idx = encoded,max_new_tokens = 50,context_size = context_size)
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()





#======================================================
#辅助函数
#======================================================
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




# ==============================================================================================
# ==============================================================================================
# 主函数
# ==============================================================================================
# ==============================================================================================


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

    tokenizer = tiktoken.get_encoding("gpt2")
    

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
    train_loader = gd.create_dataloader_v1(
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



    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 10
    train_losses,val_losses,tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs = num_epochs,
        eval_freq = 5,
        eval_iter = 5,
        start_context = "Every effort moves you",
        tokenizer = tokenizer
    )

    

