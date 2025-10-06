import torch 
from torch.utils.data import Dataset,DataLoader
import dataset_example as de 

class ToyDataset(Dataset):
    """ 自定义数据集"""
    
    def __init__(self,X,y):
        self.features = X 
        self.labels = y 
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x,one_y
    
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(de.X_train,de.y_train)
test_ds = ToyDataset(de.X_test,de.y_test)

torch.manual_seed(123)
train_loader = DataLoader(
    dataset = train_ds,
    batch_size = 2, # 批次大小
    shuffle = True, # 打乱数据
    num_workers = 0, # 多进程加载数据
   # drop_last = True # 丢弃最后一个批次
)

test_loader = DataLoader(
    dataset = test_ds,
    batch_size = 2,
    shuffle = False,
    num_workers = 0,
    drop_last = True
)


# 在训练循环中使用
'''
for batch_idx, (samples, labels) in enumerate(train_loader):
    print(f"批次 {batch_idx}:")
    print(f"  样本形状: {samples.shape}") # (batch_size, 特征数)
    print(f"  标签形状: {labels.shape}") # (batch_size,)
'''
    # ... 将数据送入模型进行训练 ...