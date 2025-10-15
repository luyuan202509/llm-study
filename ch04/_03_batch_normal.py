'''
层归一化的介绍：
实现层归一化,以提高神经网络训练的稳定性和效率,
层归一化是在神经网络中将输入数据进行归一化的一种方法。层归一化是在神经网络中的每一层都进行归一化，
而不是在神经网络中的每一个节点都进行归一化。层归一化的主要作用是提高神经网络的训练效率和稳定性。
思想是调整每一层的输入数据，使其满足均值为0，方差为1。在 GPT-2 和当前的 Transformer 架构中,层 归一化通常在多头注意力模块的前后进行

'''
import torch 
import torch.nn as nn
from torch.nn.modules import LayerNorm


torch.manual_seed(123)
batch_example = torch.randn(2,5)

layer = nn.Sequential(
    nn.Linear(5,6),nn.ReLU()
)
# 经过xian积层和激活层
out = layer(batch_example)

print(f"输出形状：{out.shape}")
print(f"输出：{out}")
print(f"输出均值：{out.mean(dim=-1, keepdim = True)}")
print(f"输出方差：{out.std(dim=-1)}")

var = out.var(dim=-1, keepdim=True)   # 均值
mean = out.mean(dim=-1, keepdim=True) # 方差
print("Mean:\n", mean) 
print("Variance:\n", var)

#===============================================================
#== 归一化
#===============================================================
print("\n====归一化========================================================")

out_norm  = (out - mean) / torch.sqrt(var)  # 减去均值，结果除以方差的平方根
mean = out_norm.mean(dim=-1, keepdim=True)

var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm) 
print("Mean:\n", mean) 
print("Variance:\n", var)

# 关闭科学计数法
print("====关闭科学计数法===========================================================")
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean) 
print("Variance:\n", var)


print("====测试DummyLayerNorm GPT模块中的层归一化===========================================================")

from _01_dummyGPT_model import DummyLayerNorm
ln = DummyLayerNorm(emb_dim = 5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)

print("Mean:\n", mean)
print("Variance:\n", var)


