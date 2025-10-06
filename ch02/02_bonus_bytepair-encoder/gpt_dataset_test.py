import os 
from gpt_dataset import GPTDataset as gd
parenpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.join(parenpath,'the-verdict.txt')

"""
创建数据集-多批次
"""

with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()

    dataloader = gd.create_dataloader_v1(raw_text,
                                      batch_size=1,
                                      max_length=4,
                                      stride = 1,
                                      shuffle = False
                                      )
    
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)
    third_batch = next(data_iter)
    print(third_batch)