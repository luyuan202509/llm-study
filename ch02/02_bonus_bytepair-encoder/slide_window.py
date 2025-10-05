import os,sys 
import tiktoken 
""" 滑动窗口提取 输入-目标对 """


paren_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file = os.path.join(paren_path,'the-verdict.txt')

tokenizer =   tiktoken.get_encoding("gpt2")

with open(file,'r',encoding="utf-8") as f:
    raw_text = f.read()
    enc_text = tokenizer.encode(raw_text)
    print("Total number of character:",len(enc_text))

    enc_sample = enc_text[50:]

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(x)
    print(y)

    print("================================================================================")

    # 下一单词预测任务
    for i in range(1,context_size +1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)

    print("================================================================================")
    for i in range(1,context_size +1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))