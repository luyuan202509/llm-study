'''张量的操作'''

import torch

tensor2d = torch.tensor([[1,2,3],[4,5,6]])
print(tensor2d.shape)

tensor2d = tensor2d.reshape(6,1)
print(tensor2d.shape)
# 更常用
tensor2d = tensor2d.view(3,2)
print(tensor2d.shape)

print('==转置====================================================================')
tensor2d = torch.tensor([[1,2,3],[4,5,6]])
print(tensor2d)
tensor2d = tensor2d.t()
print(tensor2d) 

print('==矩阵相乘====================================================================')
tensor2d = torch.tensor([[1,2],
                         [3,4],
                         [8,3]])
tensor2d1 = torch.tensor([[5,6],
                          [7,8]])
print(torch.mm(tensor2d,tensor2d1))
result = torch.matmul(tensor2d,tensor2d1)
print(result)
result3 = tensor2d.matmul(tensor2d1)
print(result3)
result4 = tensor2d @ tensor2d1
print(result4)
