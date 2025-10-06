import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# new import 
import os
import platform
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if platform.system() == 'Darwin':  # Mac OS
        # Use gloo backend for Mac MPS
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    elif platform.system() == 'Windows':
        os.environ["USE_LIBUV"] = "0"
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:  # Linux
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

class ToyDataset(Dataset):
    """ 自定义数据集"""
    def __init__(self, X, y):
        self.features = X 
        self.labels = y 
    
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
    
    def __len__(self):
        return len(self.features)

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

def compute_accuracy(model, data_loader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0
    
    for idx, (features, labels) in enumerate(data_loader):
        features, labels = features.to(device), labels.to(device)
        
        with torch.no_grad():
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct/total_examples).item()  # Return as Python float

def prepare_dataset(use_distributed=False):
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False if use_distributed else True,  # 如果使用分布式则不打乱
        pin_memory=True,
        drop_last=True if use_distributed else False,
        sampler=DistributedSampler(train_ds) if use_distributed else None
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False
    )

    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    # 只有在多设备情况下才初始化分布式训练
    use_distributed = world_size > 1
    
    if use_distributed:
        ddp_setup(rank, world_size)  # 初始化进程组
    
    # 设置设备
    if platform.system() == 'Darwin':
        device = torch.device(f'mps:{rank}' if torch.backends.mps.is_available() else 'cpu')
    elif torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
        
    train_loader, test_loader = prepare_dataset(use_distributed=use_distributed)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    # 只有在多设备情况下才使用DDP
    if use_distributed:
        # MPS上不使用device_ids参数
        if platform.system() == 'Darwin':
            model = DDP(model)  # 不指定device_ids
        else:
            model = DDP(model, device_ids=[rank])
    
    for epoch in range(num_epochs):
        # 只有在分布式训练时才设置epoch
        if use_distributed and hasattr(train_loader, 'sampler'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练模式
        model.train() if not use_distributed else model.module.train()
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # LOGGING
            prefix = f"[Rank{rank}]" if use_distributed else ""
            print(f"{prefix} Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    # 预测模式
    model.eval() if not use_distributed else model.module.eval()
    try:
        train_acc = compute_accuracy(model, train_loader, device)
        print(f"[Rank{rank}] Train Acc: {train_acc:.2f}") 
        test_acc = compute_accuracy(model, test_loader, device)   
        print(f"[Rank{rank}] Test Acc: {test_acc:.2f}")
        
        # Save model on rank 0 only
        if rank == 0:
            # 如果使用DDP，保存model.module.state_dict()，否则保存model.state_dict()
            state_dict = model.module.state_dict() if use_distributed else model.state_dict()
            torch.save(state_dict, "model_ddp.pth")
            print(f"[Rank{rank}] Model saved successfully")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    # 只有在使用分布式时才销毁进程组
    if use_distributed:
        destroy_process_group()

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", torch.backends.mps.is_available())
    
    if torch.cuda.is_available():
        print("Number of CUDA devices available:", torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        print("Number of MPS devices available:", torch.mps.device_count())
    else:
        print("Using CPU")

    torch.manual_seed(123)
        
    num_epochs = 3
    
    # Determine world size based on available hardware
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        world_size = torch.mps.device_count()
    else:
        world_size = 1
    
    print(f"World size: {world_size}")
    
    # 对于单设备情况，直接运行而不是使用mp.spawn
    if world_size <= 1:
        print("Running on single device without multiprocessing")
        main(0, max(1, world_size), num_epochs)

    else:
        print(f"Running on {world_size} devices with multiprocessing")
        # MPS fallback for distributed training
        if platform.system() == 'Darwin':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)