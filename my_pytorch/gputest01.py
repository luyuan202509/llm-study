import torch 

tensor1 = torch.tensor([1.,2.,3.])
tensor2 = torch.tensor([4.,5.,6.])
print(tensor1+tensor2)

'''
计算张量同时在一个divice上，否则会报错 都在cpu或都在gpu 进行计算
'''
tensor1 = tensor1.to("mps")
tensor2 = tensor2.to("mps")
print(tensor1+tensor2)


