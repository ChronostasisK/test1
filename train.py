from transformers import BertTokenizer, BertModel, BertConfig
import torch
import numpy as np

hashmap = {}   #定义存储标签向量的字典

data=open("C:\\Users\\dell\\Desktop\\Recommend_system\\data\\分词标签.txt",encoding='utf-8')     #读取标签txt文件

tokenizer = BertTokenizer.from_pretrained("./bert-hgd-base")
config = BertConfig.from_pretrained('./bert-hgd-base')
model = BertModel.from_pretrained("./bert-hgd-base", config=config)      #加载BERT模型

for key in data:                   #逐行遍历标签文件
    res = []                       #定义当前标签中每个字向量值的存储列表
    for inputs in key:             #逐字作为BERT输入
        output = model(torch.tensor(tokenizer.encode(inputs)).unsqueeze(0))
        res.append(output.last_hidden_state[0, 1])

    K = res[0].detach().numpy()
    for i in range(1, len(res)):
        K += res[i].detach().numpy()
    print(K)
    hashmap[key] = K
    print(key)
#
#     # param = np.array(K.last_hidden_state[0, 1]).detach().numpy()
#     # print(param)

A,B=[],[]
for i,j in zip(hashmap.keys(),hashmap.values()):
    A.append(i[:-1])
    B.append(j)
a=np.array(A)
b=np.array(B)
print(a,b)
np.savez('bert_table_embed-base', a,b)


