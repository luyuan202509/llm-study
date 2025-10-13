"""
GPT-2 规格的多头注意力模块
基于最小的 GPT-2 模型配置
"""

import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

def create_gpt2_multihead_attention():
    """
    创建符合 GPT-2 规格的多头注意力模块
    
    GPT-2 Small 模型规格：
    - 注意力头数：12 个
    - 嵌入维度：768
    - 上下文长度：1024
    """
    
    # GPT-2 模型参数
    d_in = 768          # 输入嵌入维度
    d_out = 768         # 输出嵌入维度  
    context_length = 1024  # 上下文长度
    num_heads = 12      # 注意力头数量
    dropout = 0.1       # Dropout 率
    qkv_bias = False    # GPT-2 不使用 QKV 偏置
    
    # 创建多头注意力模块
    multihead_attn = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out, 
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=qkv_bias
    )
    
    return multihead_attn

def test_gpt2_attention():
    """测试 GPT-2 多头注意力模块"""
    
    # 创建模块
    gpt2_attention = create_gpt2_multihead_attention()
    
    # 打印模块信息
    print("GPT-2 多头注意力模块配置：")
    print(f"注意力头数：{gpt2_attention.num_heads}")
    print(f"输入/输出维度：{gpt2_attention.d_out}")
    print(f"每个头的维度：{gpt2_attention.head_dim}")
    print(f"上下文长度：{gpt2_attention.mask.shape[0]}")
    print(f"总参数量：{sum(p.numel() for p in gpt2_attention.parameters()):,}")
    
    # 创建测试输入
    batch_size = 2
    seq_len = 512  # 测试序列长度
    x = torch.randn(batch_size, seq_len, 768)
    
    print(f"\n测试输入形状：{x.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = gpt2_attention(x)
    
    print(f"输出形状：{output.shape}")
    print(f"输入输出维度匹配：{x.shape == output.shape}")
    
    # 验证每个头的维度
    expected_head_dim = 768 // 12  # 64
    print(f"每个头的维度验证：{gpt2_attention.head_dim} == {expected_head_dim}")
    
    return gpt2_attention

if __name__ == "__main__":
    # 运行测试
    attention_module = test_gpt2_attention()
    
    print("\n✅ GPT-2 多头注意力模块创建成功！")
    print("配置符合最小的 GPT-2 模型规格")
