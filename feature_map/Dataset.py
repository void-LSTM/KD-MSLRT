
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import pickle
import gzip
import cv2
import os
import json
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from utils import video_augmentation
import pandas as pd
# 将图像调整为224×224尺寸并归一化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform1=transforms.Compose([
    transforms.Resize(size=[224,224]),]
    # transforms.AugMix(),
    # transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomCrop(224)]
)
transform2=transforms.Compose([
    transforms.Resize(size=[224,224])]
)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
transform_vedio = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
def shuffle_img(input):
    C,H,W=input.shape
    input=input.reshape(C,-1).transpose(1,0)
    np.random.shuffle(input)
    input=input.transpose(1,0).reshape(3,H,W)
    return input

def repalce_img(input):
    C,H_SHAPE,W_SHAPE=input.shape
    ratio = np.random.randint(0,10)/10
    size = (C,H_SHAPE,W_SHAPE)
    arr = np.random.choice([0, 1], size=size, p=[ratio, 1-ratio]).astype("bool")
    noise = np.random.choice(256, size=size)
    input=input*arr+noise*~arr
    return input.astype(np.float64) 


def random_shuffle(input):
    
    A=np.random.randint(1, 3)
    # A=2
    if A==0:
        input=shuffle_img(input)

    elif A==1:
        input=repalce_img(input)
    elif A==2:
        input=input
    # elif A==3:
    #     input=repalce_img(input)
    #     input=shuffle_img(input)

    #label=[] 输出多个，然后pad
            
    return input

def get_tensor_from_video2(video_path):
    """
    :param video_path: 视频文件地址
    :return: pytorch tensor
    """

    base_path='D:\\phoenix\\phoenix-2014.v3.tar\\phoenix2014-release\\phoenix-2014-multisigner\\features\\fullFrame-210x260px\\train\\'
    frames_list = []
    frames_label=[]
    count=0
    base_path = os.path.join(base_path, video_path)
    files = os.listdir(base_path)
    for path in files:
        full_path = os.path.join(base_path, path)
        frame=cv2.imread(full_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    vid, label = transform_vedio(frames_list, None, None)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)
    left_pad = 0
    last_stride = 1
    total_stride = 1
    kernel_sizes = ['K5', "P2", 'K5', "P2"]
    for layer_idx, ks in enumerate(kernel_sizes):
        if ks[0] == 'K':
            left_pad = left_pad * last_stride 
            left_pad += int((int(ks[1])-1)/2)
        elif ks[0] == 'P':
            last_stride = int(ks[1])
            total_stride = total_stride * last_stride

    max_len = vid.size(1)
    video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
    right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
    max_len = max_len + left_pad + right_pad
    vid = torch.cat(
        (
            vid[0,0][None].expand(left_pad, -1, -1, -1),
            vid[0],
            vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1, -1, -1),
        )
        , dim=0)   
    # if count<475:image
    #     for i in range(475-count):
    #         frames_list.append(torch.zeros((3,224,224)))
    #         frames_label.append(torch.zeros((3,224,224)))

    # 注意：此时result_frames组成的维度为[视频帧数量，宽，高，通道数]
    return vid,frames_label,video_length,video_path

class MyDataset2(Dataset):
    def __init__(self, train_path):
        super(MyDataset2, self).__init__()
        self.path=[]
        self.label=[]
        train_txt=pd.read_csv(train_path,encoding='gb18030',encoding_errors='ignore')['id|folder|signer|annotation']
        for i in train_txt:
            self.path.append(i.split('|')[1][:-6])
            self.label.append(i.split('|')[3])
        dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\feature_map\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
        gloss_dict = np.load(dict_path, allow_pickle=True).item()
        gloss_dict= dict((v[0], k) for k, v in gloss_dict.items())    
        
        ORD2CHAR = {j:i for i,j in gloss_dict.items()}
        texts_list=[]
        for i in self.label:
            sentence_list=[]
            for word in i.split(' '):
                sentence_list.append(word)
            texts_list.append(sentence_list)

        def convert_tokens_to_ids(vocab, tokens): # 输入为词表，和要转化的 text
            wids = [] # 初始化一个空的集合，用于存放输出
            #tokens = text.split(" ") # 将传入的 text 用 空格 做分割，变成 词语字符串 的列表
            for token in tokens: # 每次从列表里取出一个 词语
                wid = vocab.get(token, None)
                wids.append(wid)
            return wids
        tokens_ids = []
        for item in texts_list:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item) # 获得组成句子的 词语 的 ID 列表
            tokens_ids.append(item_ids)
        self.label=tokens_ids
            
    def __len__(self):
        return len(self.path)
 
    def __getitem__(self, index): 
        path= self.path[index]
        label=self.label[index]
        
        image,image_label,video_length,temp_path=get_tensor_from_video2(path)
        return  torch.Tensor(image), torch.Tensor(image_label), torch.Tensor(label),temp_path,video_length