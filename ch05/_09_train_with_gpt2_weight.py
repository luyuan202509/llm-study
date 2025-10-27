from gpt_download import download_and_load_gpt2
settings,params = download_and_load_gpt2(model_size="124M",models_dir="models")
import tiktoken

from previous_chapters import GPTModel,generate
import torch
import numpy as np

#print("Settings:",settings)
#print("Parameter dictionnary keys :",params.keys())

#print(params["wte"])
##print("Token embedding weight tensor dimensions:", params["wte"].shape)

#======================================================
#辅助函数
#======================================================
def text_to_token_ids(text,tokenizer):
    '''将文本转换为词元 ID 序列'''
    encoded = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    '''将词元 ID 序列转换为文本'''
    flat = token_ids.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text   


def load_weights_into_gpt(gpt,params):
    """将GPT-2权重加载到我们的模型"""
    # 位置嵌入
    gpt.pos_emb.weight.data = torch.tensor(params["wpe"], dtype=torch.float32)
    # Token嵌入
    gpt.tok_emb.weight.data = torch.tensor(params["wte"], dtype=torch.float32)

    for b in range(len(params["blocks"])):
        # 注意力层
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight.data = torch.tensor(q_w.T, dtype=torch.float32)
        gpt.trf_blocks[b].att.W_key.weight.data = torch.tensor(k_w.T, dtype=torch.float32)
        gpt.trf_blocks[b].att.W_value.weight.data = torch.tensor(v_w.T, dtype=torch.float32)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        if gpt.trf_blocks[b].att.W_query.bias is not None:
            gpt.trf_blocks[b].att.W_query.bias.data = torch.tensor(q_b, dtype=torch.float32)
        if gpt.trf_blocks[b].att.W_key.bias is not None:
            gpt.trf_blocks[b].att.W_key.bias.data = torch.tensor(k_b, dtype=torch.float32)
        if gpt.trf_blocks[b].att.W_value.bias is not None:
            gpt.trf_blocks[b].att.W_value.bias.data = torch.tensor(v_b, dtype=torch.float32)

        # 输出投影
        gpt.trf_blocks[b].att.out_proj.weight.data = torch.tensor(
            params["blocks"][b]["attn"]["c_proj"]["w"].T, dtype=torch.float32)
        if gpt.trf_blocks[b].att.out_proj.bias is not None:
            gpt.trf_blocks[b].att.out_proj.bias.data = torch.tensor(
                params["blocks"][b]["attn"]["c_proj"]["b"], dtype=torch.float32)

        # 前馈网络
        gpt.trf_blocks[b].ff.layers[0].weight.data = torch.tensor(
            params["blocks"][b]["mlp"]["c_fc"]["w"].T, dtype=torch.float32)
        gpt.trf_blocks[b].ff.layers[0].bias.data = torch.tensor(
            params["blocks"][b]["mlp"]["c_fc"]["b"], dtype=torch.float32)
        gpt.trf_blocks[b].ff.layers[2].weight.data = torch.tensor(
            params["blocks"][b]["mlp"]["c_proj"]["w"].T, dtype=torch.float32)
        gpt.trf_blocks[b].ff.layers[2].bias.data = torch.tensor(
            params["blocks"][b]["mlp"]["c_proj"]["b"], dtype=torch.float32)

        # 层归一化
        gpt.trf_blocks[b].norm1.scale.data = torch.tensor(
            params["blocks"][b]["ln_1"]["g"], dtype=torch.float32)
        gpt.trf_blocks[b].norm1.shift.data = torch.tensor(
            params["blocks"][b]["ln_1"]["b"], dtype=torch.float32)
        gpt.trf_blocks[b].norm2.scale.data = torch.tensor(
            params["blocks"][b]["ln_2"]["g"], dtype=torch.float32)
        gpt.trf_blocks[b].norm2.shift.data = torch.tensor(
            params["blocks"][b]["ln_2"]["b"], dtype=torch.float32)

    # 最终层归一化
    gpt.final_norm.scale.data = torch.tensor(params["g"], dtype=torch.float32)
    gpt.final_norm.shift.data = torch.tensor(params["b"], dtype=torch.float32)
    
    # 输出头（与token embedding共享权重）
    gpt.out_head.weight.data = torch.tensor(params["wte"], dtype=torch.float32)


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25}
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = GPT_CONFIG_124M.copy() 
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias":True})
    
   # print(NEW_CONFIG)
    gpt = GPTModel(NEW_CONFIG)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")


    torch.manual_seed(123)
    # 获取输入并移到正确的设备
    idx = text_to_token_ids("Every effort moves you", tokenizer)
    idx = idx.long().to(device)  # 确保是 Long 类型并移到正确的设备
    
    token_ids = generate(
        model=gpt,
        idx=idx,
        max_new_tokens=50,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=0.8
    )

    print(f'output text：\n {token_ids_to_text(token_ids,tokenizer)}')




