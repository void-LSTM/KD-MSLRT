# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas  as pd
import torch.nn.functional as F
import json
import random
# 自定义数据集函数
 
class MyDataset(Dataset):
    def __init__(self, train_path):
        super(MyDataset, self).__init__()
        self.root = train_path
        train_txt=pd.read_csv(train_path,encoding='gbk')
        self.label=train_txt['gloss'].values 
        dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
        gloss_dict = np.load(dict_path, allow_pickle=True).item()
        CHAR2ORD= dict((v[0], k) for k, v in gloss_dict.items())   
        self.tgt=len(CHAR2ORD)
        pad_token = '^'
        pad_token_idx = 0
        start_token='*'
        start_token_idx = len(CHAR2ORD)+1
        end_token='$'
        end_token_idx = len(CHAR2ORD)+2
        self.pad_token_idx=pad_token_idx
        CHAR2ORD[pad_token_idx] = pad_token
        CHAR2ORD[start_token_idx] = start_token
        CHAR2ORD[end_token_idx] = end_token
        ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}
        enc_input_temp=[]
        dec_input_temp=[]
        dec_output_temp=[]
        for i in self.label:
            enc_list=[]
            dec_list=[]
            decout_list=[]
            count=0
            word_list=i.split(' ')
            for word in word_list:
                if count==0:
                    dec_list.append('*')
                enc_list.append(word)
                dec_list.append(word)
                decout_list.append(word)
                if count==len(word_list)-1:
                    if len(word_list)<=32:
                        for k in range(32-len(word_list)):
                            if k==32-len(word_list)-1:
                                decout_list.append('$')
                            else:
                                dec_list.append("^")
                                decout_list.append("^")
                count+=1
            enc_input_temp.append(enc_list)
            dec_input_temp.append(dec_list)
            dec_output_temp.append(decout_list)
        def convert_tokens_to_ids(vocab, tokens): # 输入为词表，和要转化的 text
            wids = [] # 初始化一个空的集合，用于存放输出
            #tokens = text.split(" ") # 将传入的 text 用 空格 做分割，变成 词语字符串 的列表
            for token in tokens: # 每次从列表里取出一个 词语
                if token=='' or token=='__ON__'or token=='__OFF__':
                    continue
                wid = vocab.get(token)
                wids.append(wid)
                if wid==None:
                    print(tokens)
                    print(token)
                    print(wid,token)
                    print(wids)
            return wids
        enc_input=[]
        dec_input=[]
        dec_output=[]
        for item in enc_input_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            enc_input.append(item_ids)
        for item in dec_input_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            dec_input.append(item_ids)
        for item in dec_output_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            dec_output.append(item_ids)

        self.enc_input=enc_input
        self.dec_input=dec_input
        self.dec_output=dec_output
    def __len__(self):
        return len(self.enc_input)
 
    def __getitem__(self, index):
        test=self.enc_input[index]

        enc_input=self.enc_input[index]
        
        A=np.random.randint(0, 7)
        
        if A ==0:
            DEL=np.random.randint(0, len(enc_input))
            for i in range(DEL):
                index_temp=np.random.randint(0, len(enc_input))
                enc_input.pop(index_temp)
            
        elif A==1:
            random.shuffle(enc_input)
        elif A==2:

            DEL=np.random.randint(0, len(enc_input))
            for i in range(DEL):
                index_temp=np.random.randint(0, len(enc_input))
                enc_input.pop(index_temp)
            random.shuffle(enc_input)
        elif A==3:
            INSERT=np.random.randint(0, len(enc_input)//3+1)
            for i in range(INSERT):
                INSERT_index=np.random.randint(0, len(enc_input))
                INSERT_num=np.random.randint(0, self.tgt)
                enc_input.insert(INSERT_index,INSERT_num)
        elif A==4:
            INSERT=np.random.randint(0, len(enc_input)//3+1)
            for i in range(INSERT):
                INSERT_index=np.random.randint(0, len(enc_input))
                INSERT_num=np.random.randint(0, self.tgt)
                enc_input.insert(INSERT_index,INSERT_num)
            random.shuffle(enc_input)
        elif A==5:
            NUM=np.random.randint(0, len(enc_input))
            for i in range(NUM):
                index_temp=np.random.randint(0, len(enc_input))
                enc_input.pop(index_temp)
                INSERT_index=np.random.randint(0, len(enc_input))
                INSERT_num=np.random.randint(0, self.tgt)
                enc_input.insert(INSERT_index,INSERT_num)
            random.shuffle(enc_input)
        elif A==6:
            enc_input=enc_input

        enc_input = F.pad(torch.from_numpy(np.array(enc_input)), (0, int(32-len(enc_input))),'constant',value = self.pad_token_idx)
        dec_input = F.pad(torch.from_numpy(np.array(self.dec_input[index])), (0, int(32-len(self.dec_input[index]))),'constant',value = self.pad_token_idx)
        dec_output = F.pad(torch.from_numpy(np.array(self.dec_output[index])), (0, int(32-len(self.dec_output[index]))),'constant',value = self.pad_token_idx)

        
        return  enc_input, dec_input,dec_output



 
class TrainDataset(Dataset):
    def __init__(self, train_path):
        super(TrainDataset, self).__init__()
        self.root = train_path
        train_txt=pd.read_csv(train_path,encoding='gbk')
        self.input=train_txt['pred'].values 
        self.label=train_txt['gt'].values 
        dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
        gloss_dict = np.load(dict_path, allow_pickle=True).item()
        CHAR2ORD= dict((v[0], k) for k, v in gloss_dict.items())   
        self.tgt=len(CHAR2ORD)
        pad_token = '^'
        pad_token_idx = 0
        start_token='*'
        start_token_idx = len(CHAR2ORD)+1
        end_token='$'
        end_token_idx = len(CHAR2ORD)+2
        self.pad_token_idx=pad_token_idx
        CHAR2ORD[pad_token_idx] = pad_token
        CHAR2ORD[start_token_idx] = start_token
        CHAR2ORD[end_token_idx] = end_token
        ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}
        enc_input_temp=[]
        dec_input_temp=[]
        dec_output_temp=[]
        for i in self.label:
            dec_list=[]
            decout_list=[]
            count=0
            word_list=i.split(' ')
            for word in word_list:
                if count==0:
                    dec_list.append('*')
                dec_list.append(word)
                decout_list.append(word)
                if count==len(word_list)-1:
                    if len(word_list)<=32:
                        for k in range(32-len(word_list)):
                            if k==32-len(word_list)-1:
                                decout_list.append('$')
                            else:
                                dec_list.append("^")
                                decout_list.append("^")
                count+=1
            dec_input_temp.append(dec_list)
            dec_output_temp.append(decout_list)

        for i in self.input:
            enc_list=[]
            count=0
            word_list=i.split(' ')
            for word in word_list:
                enc_list.append(word)
                if count==len(word_list)-1:
                    if len(word_list)<=32:
                        for k in range(32-len(word_list)):
                            enc_list.append("^")
                count+=1
            enc_input_temp.append(enc_list)
            
        def convert_tokens_to_ids(vocab, tokens): # 输入为词表，和要转化的 text
            wids = [] # 初始化一个空的集合，用于存放输出
            #tokens = text.split(" ") # 将传入的 text 用 空格 做分割，变成 词语字符串 的列表
            for token in tokens: # 每次从列表里取出一个 词语
                if token=='':
                    continue
                wid = vocab.get(token)
                wids.append(wid)
                if wid==None:
                    print(tokens)
                    print(token)
                    print(wid,token)
                    print(wids)
            return wids
        enc_input=[]
        dec_input=[]
        dec_output=[]
        for item in enc_input_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            enc_input.append(item_ids)
        for item in dec_input_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            dec_input.append(item_ids)
        for item in dec_output_temp:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            dec_output.append(item_ids)

        self.enc_input=enc_input
        self.dec_input=dec_input
        self.dec_output=dec_output
    def __len__(self):
        return len(self.enc_input)
 
    def __getitem__(self, index):
        return  torch.Tensor(self.enc_input[index]), torch.Tensor(self.dec_input[index]), torch.Tensor(self.dec_output[index])

