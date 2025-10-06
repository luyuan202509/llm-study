import re

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {v:k for k,v in self.str_to_int.items()}
    
    def encode(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
           item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[item] for item in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join ([self.int_to_str[i] for i in ids])
        #  \s 代表任何“空白字符”
        text = re.sub(r'\s+([,.?!"()\'])', r'\1',text)
        return text 
    

if __name__ == "__main__":
 #   tokenizer = SimpleTokenizerV1(de.vocab)
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
