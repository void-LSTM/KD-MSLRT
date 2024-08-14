
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from torchvision import transforms, datasets, models
import time
import numpy as np
from datasets import *
from tqdm import tqdm
import json
from torch.utils.data.sampler import SubsetRandomSampler
from WarmupLrScheduler import GradualWarmupScheduler
from torch.optim import lr_scheduler
from transformer import Transformer
from WER import wer

def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        if characters[str_index[i].item()] =='^' or characters[str_index[i].item()] =='$':
            break
        if str_index[i] != 1203 and str_index[i] != 1204:
            char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list),char_list
if __name__ == "__main__":
    dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    CHAR2ORD= dict((v[0], k) for k, v in gloss_dict.items())   
    pad_token = '^'
    pad_token_idx = 0
    start_token='*'
    start_token_idx = len(CHAR2ORD)+1
    end_token='$'
    end_token_idx = len(CHAR2ORD)+2
    CHAR2ORD[pad_token_idx] = pad_token
    CHAR2ORD[start_token_idx] = start_token
    CHAR2ORD[end_token_idx] = end_token
    ORD2CHAR = {i:j for i,j in CHAR2ORD.items()}
    out_dim=len(ORD2CHAR)
    ValDataset=TrainDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\xjtlu_out.csv")
    Val_loader =DataLoader(ValDataset, batch_size=1,
    num_workers=0, shuffle=False)#
    

    model = Transformer(out_dim).cuda()
    checkpoint = torch.load('correct_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model2 = Transformer(len(CHAR2ORD)).cuda()
    checkpoint = torch.load('correct_model2.pth')
    model2.load_state_dict(checkpoint['model_state_dict'])

    pred_all=[]
    TOTAL_WER=0
    for enc_inputs, dec_inputs, dec_outputs in Val_loader:

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.long().cuda(), dec_inputs.long().cuda(), dec_outputs.long().cuda()
        outputs, out = model(enc_inputs, dec_inputs)
                                                        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        # enc_inputs2=out.detach().data.max(2, keepdim=True)[1].reshape(enc_inputs.shape[0],enc_inputs.shape[1])

        # outputs2, out2 = model2(enc_inputs2, dec_inputs)

        input = enc_inputs[0].contiguous().view(-1)
        
        pred1 = out[0].data.max(1, keepdim=True)[1].contiguous().view(-1)

        
        gt=dec_outputs[0].contiguous().view(-1)
        ori,ori_list=decode_out(input, ORD2CHAR)
        pred,pred_list=decode_out(pred1, ORD2CHAR)
        gt,gt_list=decode_out(gt, ORD2CHAR)
        pred_all.append([pred,gt])
        # gt_list.insert(0,'__ON__')
        # gt_list.append('__OFF__')
        # pred_list.insert(0,'__ON__')
        # pred_list.append('__OFF__')
        # ori_list.insert(0,'__ON__')
        # ori_list.append('__OFF__')
        WER=wer(r=gt_list,
        h=pred_list)
        TOTAL_WER+=WER
        # print(' ori:',ori_list)
        if WER==1:
            print(' pred:',pred_list)
            print(' gt:',gt_list)
    print('average wer: ',TOTAL_WER/len(Val_loader))
    columns_train=['pred','gt']
    df1=pd.DataFrame(data=pred_all,columns=columns_train)
    df1.to_csv('xjtlu_WER.csv',mode='a')


    
