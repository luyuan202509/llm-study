import os,sys
import tiktoken 

paren_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.join(paren_path,'the-verdict.txt')
print("Parent path:",file)

tokenizer =   tiktoken.get_encoding("gpt2")
with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()
    
    enc_text = tokenizer.encode(raw_text)
    print("Total number of character:",len(enc_text))