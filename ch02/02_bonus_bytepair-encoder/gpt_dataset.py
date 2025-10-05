import torch 
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        '''
        text: 输入文本
        tokenizer: 分词器
        max_length: 最大长度
        stride: 步长
        '''

        self.input_ids = []
        self.target_ids = []

        # 对文本进行分词
        token_ids = tokenizer.encode(text) 
        
        # 
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

    def create_dataloader_v1(text,batch_size=4,max_length=512,stride=128,drop_last = True,num_workers=0,shuffle=True):
        # 初始化分词器
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataset(text,tokenizer,max_length,stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        return dataloader
