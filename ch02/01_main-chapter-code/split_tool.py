import re

text = "Hello, world. This, is a test."

# 使用你注释中的思路 - 匹配空白字符和单引号
#regular = re.compile(r"(\s+|[^\w\s])") 
regular = re.compile(r"([,.]|\s)") 

parts = regular.split(text)
# print("使用    正则:", parts)

for idx,part in enumerate(parts):
    print("使用正则 ",idx,":",part)
