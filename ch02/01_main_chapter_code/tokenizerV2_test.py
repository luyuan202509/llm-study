import os,sys
import token 
from SimpleTokenizerV2 import SimpleTokenizerV2
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

   #分词器实例化
    tokenizer = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1,text2))
    print("原词:",text)

    
    # 词元化
    ids = tokenizer.encode(text)
    print("词ID：",ids)
    
    # 反词元化
    decode_text = tokenizer.decode(ids)
    print("反词元的词：",decode_text)

    