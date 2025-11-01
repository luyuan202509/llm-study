import urllib.request 
import zipfile 
import os 
from pathlib import Path 
import pandas as pd
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

    
    


    

    