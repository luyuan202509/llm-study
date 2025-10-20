import tiktoken  
import torch 
from previous_chapters import generate_text_simple,GPTModel

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


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False        # Query-Key-Value bias
    }

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    model = GPTModel(GPT_CONFIG_124M)
    
    token_ids = generate_text_simple(model= model,
                                     idx = text_to_token_ids(start_context,tokenizer),
                                     max_new_tokens = 10,
                                     context_size = GPT_CONFIG_124M["context_length"])
    
    print(f'token_ids: {token_ids}')
    print(f'output text:{token_ids_to_text(token_ids,tokenizer)}')
