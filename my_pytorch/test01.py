import  torch

# 创建张量
tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1,2,3])
tensor2d = torch.tensor([[1,2,3],[4,5,6]])
tensor3d = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# 浮点数
floatvec = torch.tensor([1.0,2.0,3.0])

print(tensor0d)
print(tensor1d)
print(tensor2d)
print(tensor3d)
print(floatvec)
print(tensor0d.shape)
print(tensor1d.shape)
print(tensor2d.shape)
print(tensor3d.shape)
print(floatvec.shape)

print('============================================================================')
# 张量数据类型
print(tensor0d.dtype)
print(tensor1d.dtype)
print(tensor2d.dtype)
print(tensor3d.dtype)
print(floatvec.dtype)
print(tensor0d.device)

print('==修改数据精度====================================================================')
tensor0d = tensor0d.to(torch.float32)
print(tensor0d.dtype)
print(tensor0d)


