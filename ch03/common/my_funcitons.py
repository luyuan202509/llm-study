"""公共函数"""
import os
import torch 

def get_file_path(file_name):
    """获取文件路径"""
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_path, file_name)
    return file_path

# 归一化
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()
