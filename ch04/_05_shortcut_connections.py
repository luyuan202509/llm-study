'''
跳跃连接”或“残差连接
让我们讨论一下快捷连接(也称为“跳跃连接”或“残差连接”)的概念。
快捷连接最初用 于计算机视觉中的深度网络(特别是残差网络),
目的是缓解梯度消失问题。梯度消失问题指的 是在训练过程中,梯度在反向传播时逐渐变小,导致早期网络层难以有效训练

'''
print("====前馈神经网络 GPT模块中重要模块===========================================================")

import torch
from torch import nn
from _01_dummyGPT_model import GELU

class ExampleDeepNeuralNetwork(nn.Module):
    """
   快捷连接(也称为“跳跃连接”或“残差连接”)
    """
    def __init__(self,layer_sizes,use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        # 5层网络
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0],layer_sizes[1]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1],layer_sizes[2]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2],layer_sizes[3]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3],layer_sizes[4]),GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4],layer_sizes[5]),GELU()),
        ])
       
    def forward(self,x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else :
                x = layer_output
        return x

if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]
    simple_inputs = torch.tensor([[1.,0.,-1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes,use_shortcut=False)


    def print_gradients(model,x):
        outputs = model(x)
        targets = torch.tensor([[1.]])
        loss = nn.MSELoss()
        loss = loss(outputs,targets)
        loss.backward()

        for name,param in model.named_parameters():
        #    if 'weight' in name:
               # print(f'{name} has a gradient mean of {param.grad.abs().mean().item()}')
            print(param)
    
    print_gradients(model_without_shortcut,simple_inputs)

