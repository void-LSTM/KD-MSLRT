
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
from dataset_test import *
from tqdm import tqdm
import json
from torch.utils.data.sampler import SubsetRandomSampler
from WarmupLrScheduler import GradualWarmupScheduler
from torch.optim import lr_scheduler
from BiLstm import * 
import copy
from ctcdecoder import *
from zhengliu import SLRModel
from pytorch_model_summary import summary
class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        # mask_probs = F.softmax(ref_logits[:, :, 1:], dim=-1).view(-1, ref_logits.shape[2] - 1)
        # mask = torch.max(mask_probs, dim=1)[0] > 0.5
        # if torch.sum(mask) != 0:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask) / torch.sum(mask)
        # else:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask)
        return loss
def custom_collate_fn(batch): 
        items = list(zip(*batch))
        items[0] = default_collate(items[0])
        items[1] = default_collate(items[1])
        B,TIM,C=items[1].shape
        T=TIM
        labels = list(items[2])
        items[2] = []
        items[3] = default_collate(items[3])
        target_lengths = torch.zeros((len(batch,)), dtype=torch.int)
        input_lengths = torch.zeros(len(batch,), dtype=torch.int) 
        for idx, label in enumerate(labels):
            # 记录每个图片对应的字符总数
            target_lengths[idx] = len(label)
            # 将batch内的label拼成一个list
            items[2].extend(label)
            # input_lengths 恒为 T
            input_lengths[idx] = T
        return items[0],items[1], torch.tensor(items[2]), target_lengths, input_lengths,items[3]

def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if str_index[i] != 0 and  str_index[i] != 1203 and str_index[i] != 1204:
            char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list)
class Config:
    def __init__(self):

        # 模型配置
        # self.lstm_hidden_size = 256
        # self.dense_hidden_size = 2048
        # self.embed_size = 2048
        # self.num_layers = 2
        self.lstm_hidden_size = 256
        self.dense_hidden_size = 512
        self.embed_size = 512
        self.num_layers = 1
if __name__ == '__main__':
    dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    gloss_dict= dict((v[0], k) for k, v in gloss_dict.items())   
    out_dim=len(gloss_dict)

    
    # Character to Ordinal Encoding Mapping   
    np.random.seed(123)
    torch.manual_seed(123)

    BATCH_SIZE = 1

    LR = 0.0001
    EPOCHS =600
    shuffle_dataset = True
    random_seed= 42
    USE_GPU = True
    if USE_GPU:
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = False
    net = BiLSTM_SA(Config(),len(gloss_dict)+1)
    net=net.to(device)
    # checkpoint = torch.load('C:\\Users\\wuxin\\Desktop\\KD_test\\kd_test.pth')
    # net.load_state_dict(checkpoint['model_state_dict'])


 
    
    optimizer = optim.Adam(filter(lambda param: param.requires_grad == True,net.parameters()),lr=LR,weight_decay=0.0)
    exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=5,verbose=False)
    my_lr_scheduler=GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=exp_lr_scheduler)


    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    kd_loss=SeqKD(T=4)
    frame=475
    
    trainDataset=MyDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\PHOENIX_parquet.csv",frame)
    devDataset=MyDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\PHOENIX_parquet_dev.csv",frame)
    validation_loader =DataLoader(devDataset, batch_size=BATCH_SIZE,
    num_workers=3, pin_memory=True,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    train_loader =DataLoader(trainDataset, batch_size=BATCH_SIZE,
    num_workers=3, pin_memory=True,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    conv_logits = np.load('conv_logits.npy', allow_pickle=True)
    conv_logits=conv_logits.item()
    sequence_logits = np.load('sequence_logits.npy', allow_pickle=True)
    sequence_logits=sequence_logits.item()
    val_loss=[]
    loop = tqdm(train_loader, desc='Train')
    count=0

    flag=0
    for epoch in range(EPOCHS):
        loss_sum=0
        loop = tqdm(train_loader, desc='Train')
        net.train()
        
        valid_epoch_loss=0
        if epoch>=00: 
            ep=0
            for i in range(40):
                net.train()
                loss_sum=0
                loop = tqdm(train_loader, desc='Train')
                count=0
                for main_feature,main_feature_label, labels, target_lengths, input_lengths,path  in loop:
                    main_feature=main_feature.float().to(device)
                    main_feature_label=main_feature_label.float().to(device)
                    net.zero_grad()
                    # data = data.view(-1, 28*28)
                    out,de_out,conv_out = net(main_feature_label)
                    T,_,_=conv_out.shape

                    conv_logits_label=torch.tensor(conv_logits[path[0]]).to(device)
                    sequence_logits_label=torch.tensor(sequence_logits[(path[0])]).to(device)


                    if count<=0:
                        pred, score, labels_p = ctc_beam_search_decode(sequence_logits_label.softmax(-1).permute(1, 0, 2)[0].detach().cpu().numpy() , beam_size=10, blank=0)

                        print('pred:',decode_out(pred[0], gloss_dict),' gt:',decode_out(labels[:target_lengths[0]].contiguous().view(-1),gloss_dict))

                        count+=1
                    ctc_conv=ctc_loss(log_probs=conv_out.log_softmax(-1), targets=labels, target_lengths=target_lengths, input_lengths=torch.tensor(T))
                    ctc_bi=ctc_loss(log_probs=out.log_softmax(-1), targets=labels, target_lengths=target_lengths, input_lengths=torch.tensor(T))
                    kd_conv=kd_loss(conv_out,conv_logits_label)
                    kd_bi=kd_loss(out,sequence_logits_label)
                    if flag==0:
                        loss_ctc = ctc_conv+ctc_bi+10*kd_bi+10*kd_conv
                    else:
                        loss_ctc = 10*kd_bi+10*kd_conv+ctc_bi*0.01
                    # loss_ctc=ctc_bi
                    total_loss=loss_ctc
                    total_loss.backward()
                    loss_sum+=float(ctc_loss(log_probs=out.log_softmax(-1), targets=labels, target_lengths=target_lengths, input_lengths=torch.tensor(T)))
                    optimizer.step()
                    # for p in net.parameters():
                    #     p.data.clamp_(-0.25, 0.25)
                    loop.set_postfix(loss_ctc = float(ctc_bi),kd_conv=float(kd_conv),kd_bi=float(kd_bi))#,kd_ve=float(kd_self))
                    
                    
                net.eval()
                valid_epoch_loss = 0
                count=0
                with torch.no_grad():
                
                    pred_all=[]
                    loop = tqdm(validation_loader, desc='Train')
                    for main_feature,main_feature_label, labels, target_lengths, input_lengths,path  in loop:
                        main_feature=main_feature.float().to(device)
                        main_feature_label=main_feature_label.float().to(device)
                        out,_,_ = net(main_feature_label)
                        T,_,_=out.shape
                        loss_ctc = ctc_loss(log_probs=out.log_softmax(-1), targets=labels, target_lengths=target_lengths, input_lengths=torch.tensor(T))
                        valid_epoch_loss+=loss_ctc
                        # pred, score, labels_p = ctc_beam_search_decode(out.softmax(-1).permute(1, 0, 2)[0].cpu().numpy() , beam_size=10, blank=0)
                        # pred_all.append([decode_out(pred[0], gloss_dict),decode_out(labels[:target_lengths[0]].contiguous().view(-1),gloss_dict)])

                        if count<=5:
                            pred, score, labels_p = ctc_beam_search_decode(out.softmax(-1).permute(1, 0, 2)[0].cpu().numpy() , beam_size=10, blank=0)
                            print('pred:',decode_out(pred[0], gloss_dict),' gt:',decode_out(labels[:target_lengths[0]].contiguous().view(-1),gloss_dict))
                            count+=1
                            # if count==10:
                            #     break
                    # columns_train=['pred','gt']
                    # df1=pd.DataFrame(data=pred_all,columns=columns_train)
                    # df1.to_csv('xjtlu_out.csv',mode='w')
                # if valid_epoch_loss/len(validation_loader)>loss_sum/len(train_loader)-0.5:
                #     flag=1
                # else:
                #     flag=0
                print('Train Epoch: {},Loss: {:.6f}'.format(ep,loss_sum/len(train_loader)),'valid Epoch: {},Loss: {:.6f}'.format(epoch,valid_epoch_loss/len(validation_loader)))
                ep+=1
                my_lr_scheduler.step(metrics=valid_epoch_loss/len(validation_loader))
                torch.save({
                    'model_state_dict': net.state_dict(),
                    }, 'kd_test.pth')

        # 训练时,每个epoch结束后获取验证集loss
        my_lr_scheduler.step(metrics=valid_epoch_loss/len(validation_loader))
        # val_loss.append(valid_epoch_loss/len(validation_loader))
        print('Train Epoch: {},Loss: {:.6f}'.format(epoch,loss_sum/len(train_loader)),'valid Epoch: {},Loss: {:.6f}'.format(epoch,valid_epoch_loss/len(validation_loader)))
        print(optimizer.state_dict()['param_groups'][0]['lr']) 
        torch.save({
            'model_state_dict': net.state_dict(),
            }, 'kd_test.pth')
    print(val_loss)

