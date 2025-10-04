import re

class SimpleTokenizerV2:
    """ 分词器""" 
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in self.str_to_int.items()}
    
    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
           #去除空白字符
           item.strip() for item in preprocessed if item.strip() #
        ]
        # 使用<|unk|> 替换没在在字典中的词元
        preprocessed = [ item if item in self.str_to_int else "<|unk|>" for item in preprocessed] 

        ids = [self.str_to_int[item] for item in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join ([self.int_to_str[i] for i in ids])
        #  \s 代表任何“空白字符”
        text = re.sub(r'\s+([,.?!"()\'])', r'\1',text)
        return text 