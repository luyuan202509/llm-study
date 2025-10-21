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

    context1 = "every effort moves"
    context2 = "I really like"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_context1 = text_to_token_ids(context1,tokenizer)
    encoded_context2 = text_to_token_ids(context2,tokenizer)

    target_text1 = " effort moves you"
    target_text2 = " really like chocolate"
    encoded_target1 = text_to_token_ids(target_text1,tokenizer)
    encoded_target2 = text_to_token_ids(target_text2,tokenizer)



    inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

    targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [1107,  588, 11311]]) #  " really like chocolate"]


    model = GPTModel(GPT_CONFIG_124M)
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
    print(probas.shape) # Shape: (batch_size, num_tokens, vocab_size)

    # 获取每个词元的概率
    token_ids =  torch.argmax(probas, dim=-1,keepdim=True)
    print(token_ids)

    print(f"targets batch1:{token_ids_to_text(targets[0],tokenizer)}")
    print(f"Output batch1:{token_ids_to_text(token_ids[0].squeeze(),tokenizer)}")
    