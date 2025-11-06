import torch 
from previous_chapters import GPTModel,load_weights_into_gpt,generate_text_simple,text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2,get_settings_and_params
from _04_prepare_dataset import SpamDataset 
from torch.utils.data import DataLoader
import time 

# 下载模型配置和权重
#settings,params = download_and_load_gpt2(model_size="124M",models_dir="models")
# 已经下载，直接加载即可，免得重复下载
settings,params = get_settings_and_params(model_size="124M",models_dir="models")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

def calc_accuracy_loader(data_loader,model,device,num_batches=None):
    """计算数据加载器的准确率"""
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is  None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i< num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # [batch_size, num_classes]
                logits
                predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size]
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break;
    
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader,model,device,num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
    for i,(input_batch,target_batch) in enumerate(data_loader):
        if i< num_batches:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches





CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_config = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_config[CHOOSE_MODEL])

#print(model)

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# 下载模型配置和权重
#settings, params = download_and_load_gpt2( model_size=model_size, models_dir="gpt2" )
# 初始化模型
model = GPTModel(BASE_CONFIG)
# 加载权重
load_weights_into_gpt(model,params)
model.eval()


text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_1,tokenizer),
    max_new_tokens = 15,
    context_size = BASE_CONFIG["context_length"]
)
print("验证模型输出：",token_ids_to_text(token_ids,tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':" 
    " 'You are a winner you have been specially" 
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_2,tokenizer),
    max_new_tokens = 23,
    context_size = BASE_CONFIG["context_length"]
)
print("分类验证模型垃圾消息分类器：",token_ids_to_text(token_ids,tokenizer))


print("\n\n============================================================================================\n  ")
# 查看模型机构
print("模型结构：",model)

## 添加分类头
torch.manual_seed(123)
num_classes = 2
#model = GPTModel(BASE_CONFIG)
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=num_classes,bias=False
)

# 使最终层归一化和最后一个 Transformer 块可训练
for param in model.final_norm.parameters():
    param.requires_grad = True
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

print("\n\n============================================================================================\n  ")
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

# 生成文本
with torch.no_grad():
    logits = model(inputs)

print("Logits dimensions:", logits.shape)
print("Logits:", logits)

print("\n\n============================================================================================\n  ")
# 只关注最后一个词元
print("Last output token:", logits[:, -1, :])

print("============================================================================================\n  ")
print("计算准确率")
train_dataset = SpamDataset(
csv_file="dataset/train.csv",
max_length=None,
tokenizer=tokenizer
) 

val_dataset = SpamDataset(
csv_file="dataset/validation.csv",
max_length=None,
tokenizer=tokenizer
) 

test_dataset = SpamDataset(
csv_file="dataset/test.csv",
max_length=None,
tokenizer=tokenizer
) 


num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
    drop_last = True
)
val_loader = DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    drop_last = True
)
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    num_workers = num_workers,
    drop_last = False
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
torch.manual_seed(123) 
train_accuracy = calc_accuracy_loader(train_loader,model,device,num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader,model,device,num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader,model,device,num_batches=10)
print(f"Train accuracy:{train_accuracy*100:0.2f}%", )
print(f"Val accuracy:{val_accuracy*100:0.2f}%", )
print(f"Test accuracy:{test_accuracy*100:0.2f}%", )

print("\n\n============================================================================================\n  ")
print("计算损失")
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader,model,device,num_batches=5)
    val_loss = calc_loss_loader(val_loader,model,device,num_batches=5)
    test_loss = calc_loss_loader(test_loader,model,device,num_batches=5)
print(f"Train loss:{train_loss:0.3f}", )
print(f"Val loss:{val_loss:0.3f}", )
print(f"Test loss:{test_loss:0.3f}", )



print("\n\n============================================================================================\n  ")
print("训练分类器模型")

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                train_loss,val_loss = evaluate_model(model,train_loader,val_loader,device,eval_iter)
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")