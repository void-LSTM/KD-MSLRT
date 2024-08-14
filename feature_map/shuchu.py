
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
from Dataset import *
from tqdm import tqdm
import json
from torch.utils.data.sampler import SubsetRandomSampler
from WarmupLrScheduler import GradualWarmupScheduler
from torch.optim import lr_scheduler
import pandas as pd
from resnet2 import MAE_video
from WER import wer
def custom_collate_fn(batch, T=190):
        items = list(zip(*batch))
        items[0] = default_collate(items[0])
        items[1] = default_collate(items[1])
        labels = list(items[2])
        items[2] = []
        target_lengths = torch.zeros((len(batch,)), dtype=torch.int)
        input_lengths = torch.zeros(len(batch,), dtype=torch.int) 
        for idx, label in enumerate(labels):
            # 记录每个图片对应的字符总数
            target_lengths[idx] = len(label)
            # 将batch内的label拼成一个list
            items[2].extend(label)
            # input_lengths 恒为 T
            input_lengths[idx] = T
        return items[0],items[1], torch.tensor(items[2]), target_lengths, input_lengths

def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if str_index[i] != len(ORD2CHAR)-1 and (not (i > 0 and str_index[i - 1] == str_index[i])):
            char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list)
if __name__ == '__main__':
    with open('D:\\science\\word.json') as json_file:
        CHAR2ORD = json.load(json_file)
    pad_token = '^'
    pad_token_idx = len(CHAR2ORD)
    CHAR2ORD[pad_token] = pad_token_idx
    ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}
    out_dim=len(ORD2CHAR)

    
    # Character to Ordinal Encoding Mapping   
    np.random.seed(123)
    torch.manual_seed(123)

    BATCH_SIZE = 1

    LR = 0.00005
    EPOCHS =170
    shuffle_dataset = True
    random_seed= 42
    USE_GPU = True
    if USE_GPU:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = False
    net = MAE_video(len(ORD2CHAR),1000)
    net=net.to(device)
    checkpoint = torch.load('./test.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(filter(lambda param: param.requires_grad == True,net.parameters()),lr=LR,weight_decay=0)
    exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=5,verbose=False)
    my_lr_scheduler=GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=exp_lr_scheduler)
   

    ctc_loss = nn.CTCLoss(blank=len(CHAR2ORD)-1, reduction='mean')
    loss_f = nn.MSELoss() 
    frame=475
    devDataset2=MyDataset2("./phoenix14t.pami0.dev.annotations_only.gzip")
    validation_loader =DataLoader(devDataset2, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=False,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    loop = tqdm(validation_loader, desc='Train')
    net.eval()
    pred=[]
    gt=[]
    valid_epoch_loss = 0
    TOTAL_WER=0
    with torch.no_grad():
        for image,image_label, labels, target_lengths, input_lengths  in loop:
            image=image.float().cuda().transpose(1,2)
            image_label=image_label.float().cuda().transpose(1,2)
            labels=labels.cuda()
            IMAGE_pred,out=net(image)
            loss_ctc = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
            valid_epoch_loss+=loss_ctc
            _, preds = out.max(2)
            pred1 = preds.transpose(1, 0)[0,:].contiguous().view(-1)
            pred.append([decode_out(pred1, ORD2CHAR),decode_out(labels[:target_lengths[0]].contiguous().view(-1),ORD2CHAR)])
            #print('pred:',decode_out(pred1, ORD2CHAR),' gt:',decode_out(labels[:target_lengths[0]].contiguous().view(-1),ORD2CHAR))
            WER=wer(r=decode_out(labels[:target_lengths[0]].contiguous().view(-1),ORD2CHAR),
            h=decode_out(pred1, ORD2CHAR))
            TOTAL_WER+=WER
        print('average wer: ',TOTAL_WER/len(validation_loader))
        print('val:',valid_epoch_loss/len(validation_loader),flush=True)
        columns_train=['pred','gt']
        df1=pd.DataFrame(data=pred,columns=columns_train)
        df1.to_csv('xjtlu_out.csv',mode='w')
    
