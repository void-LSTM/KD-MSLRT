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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
def get_tensor_from_video2(video_path):
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
    count=0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if not ret:
            break
        else:
            count+=1
    return count

with gzip.open("./phoenix14t.pami0.train.annotations_only.gzip", 'rb') as f:
    annotations = pickle.load(f)
temp=[]
for i in tqdm(annotations):
    temp.append(get_tensor_from_video2('./'+i['name']+'.mp4'))

plt.figure(figsize=(10,6), dpi=100, facecolor="w")

plt.hist(temp, bins=15, edgecolor='black')
plt.xlabel("sepal_length")
plt.ylabel("Frequency")
plt.show()
