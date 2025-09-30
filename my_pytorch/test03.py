""" 自动微分 """

import torch
import torch.nn.functional as F
from torch.autograd import grad 


# 逻辑回归的前向传播
y = torch.tensor([1.0])
x1  = torch.tensor([1.1])
W1 = torch.tensor([2.2],requires_grad=True)
b1 = torch.tensor([0.1],requires_grad=True)
z = x1 * W1 + b1 #
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a,y)
print(loss)

grad_L_w1 = grad(loss, W1, retain_graph=True)
grad_L_b1 = grad(loss, b1, retain_graph=True)

print(grad_L_w1)    
print(grad_L_b1)

loss.backward()
print(W1.grad)
print(b1.grad)

