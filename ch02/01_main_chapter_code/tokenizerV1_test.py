import os,sys
import token 
from SimpleTokenizerV1 import SimpleTokenizerV1
import re

paren_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(paren_path,'01_main_chapter_code')
file = file_path + '/the-verdict.txt'
with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:",len(raw_text))
    #print(raw_text[:99])
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) 
    print("Total number of character:",len(preprocessed)) # 包括空白字符
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("Total number of character:",len(preprocessed)) # 不包括空白字符

    # 打印前30 
    print(preprocessed[:30])

    # 词元转换为词元 ID
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print("词元ID大小:",vocab_size) # 1130

    print("============================================================================")
    # 创建词汇表字典 
    vocab = {word:i for i,word in enumerate(all_words)}
    
    tokenizer = SimpleTokenizerV1(vocab)
    
    text = """"It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

   # text2 = "Hello, do you like tea?" # 不在数据集中，会报错
    #print(tokenizer.encode(text2))

