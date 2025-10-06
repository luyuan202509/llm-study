import os,sys
import token 
from SimpleTokenizerV1 import SimpleTokenizerV1
import re

paren_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(paren_path,'01_main_chapter_code')
file = file_path + '/the-verdict.txt'
with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()
    #print(raw_text[:99])
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) 
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    
    # 词元转换为词元 ID
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>","<|unk|>"])
    vocab_size = len(all_tokens)
    #print("词元ID大小:",vocab_size) # 1132

    print("============================================================================")
    # 创建词汇表字典 
    vocab = {word:i for i,word in enumerate(all_tokens)}

    #验证词典
    for i,item in enumerate(list(vocab.items())[-5:]):
        print(i,item)
    
    tokenizer = SimpleTokenizerV1(vocab)
    
    text = """"It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
   # print(ids)

   # text2 = "Hello, do you like tea?" # 不在数据集中，会报错
    #print(tokenizer.encode(text2))

