import torch 
from previous_chapters import GPTModel,load_weights_into_gpt,generate_text_simple,text_to_token_ids, token_ids_to_text
from gpt_download import download_and_load_gpt2,get_settings_and_params


# 下载模型配置和权重
#settings,params = download_and_load_gpt2(model_size="124M",models_dir="models")
# 已经下载，直接加载即可，免得重复下载
settings,params = get_settings_and_params(model_size="124M",models_dir="models")

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")




CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

model_config = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12}, 
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}, 
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20}, 
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_config[CHOOSE_MODEL])

#print(model)

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# 下载模型配置和权重
#settings, params = download_and_load_gpt2( model_size=model_size, models_dir="gpt2" )
# 初始化模型
model = GPTModel(BASE_CONFIG)
# 加载权重
load_weights_into_gpt(model,params)
model.eval()


text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_1,tokenizer),
    max_new_tokens = 15,
    context_size = BASE_CONFIG["context_length"]
)
print("验证模型输出：",token_ids_to_text(token_ids,tokenizer))

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':" 
    " 'You are a winner you have been specially" 
    " selected to receive $1000 cash or a $2000 award.'"
)
token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids(text_2,tokenizer),
    max_new_tokens = 23,
    context_size = BASE_CONFIG["context_length"]
)
print("分类验证模型垃圾消息分类器：",token_ids_to_text(token_ids,tokenizer))
