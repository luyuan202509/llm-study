import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text  = ( "Hello, do you like tea? <|endoftext|> In the sunlit terraces" "of someunknownPlace.")
# 词元转换为词元 ID
integers = tokenizer.encode(text,allowed_special={"<|endoftext|>"})
print(integers)
# 词元 ID 转换为词元
text = tokenizer.decode(integers)
print(text)

print("============================================================================")
# 未知词汇的BPE 分词 
text1 = "Akwirw ier"
ids =  tokenizer.encode(text1)
print(ids)  
text2 = tokenizer.decode(ids)
print(text2)

