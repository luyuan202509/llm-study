import torch
import torch.nn.functional as F
from torch.utils.data import dataloader
from mul_network import NeuralNetwork
from myDataset import train_loader,test_loader

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2,num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(),lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for batch_idx,(features,labels) in enumerate(train_loader):
       logits = model(features)    # 向前传播
       loss = F.cross_entropy(logits,labels) # 计算损失
       optimizer.zero_grad()       # 清零梯度
       loss.backward()              # 反向传播
       optimizer.step()             # 更新参数
       
       print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}" 
             f" | Batch {batch_idx:03d}/{len(train_loader):03d}" 
             f" | Train Loss: {loss:.2f}")
    
# 推理模式
"""
model.eval()
with torch.no_grad():
    for features,labels in test_loader:
        outputs = model(features)
        print(outputs)
   
"""
# 计算准确率
def compute_accuracy(model,dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0    
    for indx,(features,labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits,dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct/total_examples).item()

print("===============================")
print(compute_accuracy(model,train_loader))
print(compute_accuracy(model,test_loader))