import os 
from gpt_dataset import GPTDataset as gd
parenpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.join(parenpath,'the-verdict.txt')
"""创建数据集-单批次
 """

with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()

    dataloader = gd.create_dataloader_v1(raw_text,
                                      batch_size=8,
                                      max_length=4,
                                      stride = 4,
                                      shuffle = False
                                      )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n",inputs)
    print("Targets:\n",targets)   