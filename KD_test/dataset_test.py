import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas  as pd
import torch.nn.functional as F
import json
import cv2
from tqdm import tqdm
import math
import video_augmentation
TRescale=video_augmentation.TemporalRescale(0.2, 1)
transform_vedio = video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
                
            ])
def get_tensor_from_video2(video_path,rand):
    """
    :param video_path: 视频文件地址
    :return: pytorch tensor
    """
    if not os.access(video_path, os.F_OK):
        print('测试文件不存在')
        return
    cap = cv2.VideoCapture(video_path)

    frames_list = []
    frames_label=[]
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        else:
            # 注意，opencv默认读取的为BGR通道组成模式，需要转换为RGB通道模式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
    vid, label = transform_vedio(frames_list, None, None)
    # vid=TRescale(vid,rand)
    vid = vid.float() / 127.5 - 1
    vid = vid.unsqueeze(0)
    left_pad = 0
    last_stride = 1
    total_stride = 1
    kernel_sizes = ['K3',"P2", 'K5', "P2"]
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
    cap.release()

    # 注意：此时result_frames组成的维度为[视频帧数量，宽，高，通道数]
    return vid,frames_label,video_length
def shuffle_img(input):
    L,F=input.shape
    ratio = np.random.randint(0,20)/100-0.1
    input=input.reshape(L,int(F/3),3)
    input[:,:,0]=input[:,:,0]+ratio
    input[:,:,1]=input[:,:,0]+ratio
    input=input.reshape(L,F)
    return input.astype(np.float64) 

def flip_land(input):
    L,F=input.shape
    ratio = np.random.randint(0,30)
    b = math.radians(ratio)
    size = (L,F)
    input=input.reshape(L,int(F/3),3)
    input[:,:,0]=input[:,:,0]*math.cos(b)-input[:,:,1]*math.sin(b)
    input[:,:,1]=input[:,:,0]*math.sin(b)+input[:,:,1]*math.cos(b)
    input=input.reshape(L,F)
    return input.astype(np.float64) 

def fliptr_land(input):
    L,F=input.shape
    ratio = np.random.randint(0,30)
    b = math.radians(ratio)
    size = (L,F)
    input=input.reshape(L,int(F/3),3)
    input[:,:,0]=-input[:,:,0]
    input=input.reshape(L,F)

    
    return input.astype(np.float64) 

def random_shuffle(input):
    
    A=np.random.randint(0, 5)
    # A=1
    input=flip_land(input)
    input=fliptr_land(input)
    input=shuffle_img(input)
    rand=np.random.random()
    # input=TRescale(input,rand)
    
            
    return input,rand
def column_list():
    columns=[]
    LIP = [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    POSE = [13, 15, 17, 19, 21,14, 16, 18, 20, 22]
    for j in ['x','y','z']:
        for i in ['pose_landmarks','face_landmarks','left_hand_landmarks','right_hand_landmarks']:
            if i=='pose_landmarks':
                for k in POSE:
                    columns.append(j+'_'+str(k)+'_'+i)
            elif i =='face_landmarks':
                for k in LIP:
                    columns.append(j+'_'+str(k)+'_'+i)
            else:
                for k in range(21):
                    columns.append(j+'_'+str(k)+'_'+i)
    return columns
        
class MyDataset(Dataset):
    def __init__(self, train_path,frame):
        super(MyDataset, self).__init__()
        self.root = train_path
        train_txt=pd.read_csv(train_path,encoding_errors='ignore')
        self.label=train_txt['gloss'].values 
        self.feature_paths=train_txt['path']
        self.ori_paths=train_txt['ori_path'].values 

        self.feature=[]
        self.column_list=column_list()

        dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\feature_map\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
        gloss_dict = np.load(dict_path, allow_pickle=True).item()
        gloss_dict= dict((v[0], k) for k, v in gloss_dict.items())    
        
        ORD2CHAR = {j:i for i,j in gloss_dict.items()}



        def convert_tokens_to_ids(vocab, tokens): # 输入为词表，和要转化的 text
            wids = [] # 初始化一个空的集合，用于存放输出
            #tokens = text.split(" ") # 将传入的 text 用 空格 做分割，变成 词语字符串 的列表
            for token in tokens: # 每次从列表里取出一个 词语
                if token=='':
                    continue
                wid = vocab.get(token, None)
                
                wids.append(wid)
                if wid==None:
                    print(tokens)
                    print(token)
                    print(wid,token)
                    print(wids)
            return wids
        tokens_ids = []
        for item in self.label:
            item_ids = convert_tokens_to_ids(ORD2CHAR, item.split(" ")) # 获得组成句子的 词语 的 ID 列表
            tokens_ids.append(item_ids)
        self.labels=tokens_ids
        self.frame=frame
    def __len__(self):
        return len(self.ori_paths)
 
    def __getitem__(self, index):
        # main_feature = self.feature[index]
        label = self.labels[index]
        feature_path=self.feature_paths[index]
        
        try:
            all_feature=pd.read_parquet('D:\\phoenix\\phoenix-2014.v3.tar\\phoenix2014-release\\code/'+feature_path)
            main_feature=all_feature[self.column_list].values
        except:
            print('D:\\phoenix\\phoenix-2014.v3.tar\\phoenix2014-release\\code/'+feature_path)
        # main_feature=main_feature.values
        vid,rand=random_shuffle(main_feature)
        left_pad = 0
        last_stride = 1
        total_stride = 1
        kernel_sizes = ['K5','K5', "P2", 'K5', "P2"]
        for layer_idx, ks in enumerate(kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        vid=torch.tensor(vid).unsqueeze(0)
        max_len = vid.size(1)
        video_length = torch.LongTensor([np.ceil(vid.size(1) / total_stride) * total_stride + 2*left_pad ])
        right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
        max_len = max_len + left_pad + right_pad
        vid = torch.cat(
            (
                vid[0,0][None].expand(left_pad, -1),
                vid[0],
                vid[0,-1][None].expand(max_len - vid.size(1) - left_pad, -1),
            )
            , dim=0)   
        # vid = torch.cat(
        #     (
        #         vid[0],
        #         vid[0,-1][None].expand(max_len - vid.size(1), -1),
        #     )
        #     , dim=0)   
        try:
            return torch.Tensor(main_feature),torch.Tensor(vid),torch.Tensor(label),self.ori_paths[index]
        except:
            print('error')
