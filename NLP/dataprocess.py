from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path='bert-base-chinese',  # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
    cache_dir=None,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    force_download=False,   
)
def tran(seq):
    token = tokenizer.tokenize(seq)
    ids = tokenizer.convert_tokens_to_ids(token)
    return ids
zidian = tokenizer.get_vocab()
import os
 
   
def get_filelist(dir, Filelist):
 
    newDir = dir
 
    if os.path.isfile(dir):
 
        Filelist.append(dir)

 
    elif os.path.isdir(dir):
 
        for s in os.listdir(dir):

 
            newDir=os.path.join(dir,s)
 
            get_filelist(newDir, Filelist)
 
    return Filelist
 
 
 
if __name__ =='__main__' :
 
    list = get_filelist('D:\\NLP\\news2016zh_corpus', [])
 
    print(len(list))
    temp=[]
    count=0
    temp_num=0
    for e in tqdm(list):
        with open(e, 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')       #去除文本中的换行符
                seq=tran(ann)
                if len(seq)!=0:
                    temp.append([seq])
                    count+=1
                if count==128:
                    count=0
                    name=['text']
                    data = pd.DataFrame(data = temp,index = None,columns = name)
                    PATH='D:\\NLP\\transformer-main\\data\\'+str(temp_num)+'.csv'
                    data.to_csv(PATH,mode='w')
                    temp_num+=1
                    temp=[]
                    
 



