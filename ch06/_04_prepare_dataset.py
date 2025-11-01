import urllib.request 
import zipfile 
import os 
from pathlib import Path 
import pandas as pd
import tiktoken

import torch 
from torch.utils.data import Dataset,DataLoader

class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length=None,pad_token=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data['Text']]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [ encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
        
        self.encoded_texts = [
            encoded_text + [pad_token] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded,dtype=torch.long),
                torch.tensor(label,dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


# 下载并解压垃圾邮件数据集
def download_and_unzip_spam_data(url,zip_path,extracted_path,data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path}  already exists,Skipping download and extraction")
        return 
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path,"wb") as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path,data_file_path)
    print(f"Data file downloaded and extracted to: {data_file_path}")


# 创建平衡数据集
def create_balanced_dataset(df):
    num_spam = df[df['Label']=='spam'].shape[0] # 垃圾消息数量，即行数
    ham_subset = df[df['Label']=='ham'].sample( num_spam,random_state=123)
    
    balanced_df = pd.concat([ham_subset,df[df['Label']=='spam']])
    return balanced_df

# 拆分数据集为训练集、验证集和测试集
def random_split(df,train_frac,validation_frac):
    df = df.sample(frac=1,random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df,validation_df,test_df


# 将“string”类别标签"ham"和"spam"分别转换为整数类别标签 0 和 1:
def convert_label_to_int(df):
    df["Label"] = df["Label"].map({"ham":0,"spam":1})
    return df



if __name__ == "__main__":
    extracted_path = "sms_spam_collection" 
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"
    df = pd.read_csv(data_file_path,sep='\t',header=None,names=['Label','Text'])
    
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())


    # 将“string”类别标签"ham"和"spam"分别转换为整数类别标签 0 和 1:
    balanced_df["Label"] = balanced_df["Label"].map({"ham":0,"spam":1})

    
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

    train_df.to_csv("dataset/train.csv", index=None)
    validation_df.to_csv("dataset/validation.csv", index=None)
    test_df.to_csv("dataset/test.csv", index=None)
    

    tokenizer=tiktoken.get_encoding("gpt2")


    train_dataset = SpamDataset(
    csv_file="dataset/train.csv",
    max_length=None,
    tokenizer=tokenizer
    ) 
    
    print(train_dataset.max_length)
