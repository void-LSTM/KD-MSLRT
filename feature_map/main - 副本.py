
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
from torch.optim import lr_scheduler
from resnet import * 
from WarmupLrScheduler import GradualWarmupScheduler
def custom_collate_fn(batch, T=250):
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
    with open('./word.json') as json_file:
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
    EPOCHS = 100
    shuffle_dataset = True
    random_seed= 42
    USE_GPU = True
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        device_ids = [0, 1]
        # 就这一行
        model = MAE_video(len(ORD2CHAR),512,[1, 1, 1, 1])
        checkpoint = torch.load('./test.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = nn.DataParallel(model,device_ids=device_ids)
        model=model.cuda(device=device_ids[0])
    else:
        model = MAE_video(len(ORD2CHAR),512,[1, 1, 1, 1])
        # checkpoint = torch.load('./test.pth')
        # model.load_state_dict(checkpoint['model_state_dict'])
        model=model.cuda()

    frame=475
    

    optimizer = optim.Adam(filter(lambda param: param.requires_grad == True,model.parameters()),lr=LR,weight_decay=0.0001)
    exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=5,verbose=False)
    my_lr_scheduler=GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=20, after_scheduler=exp_lr_scheduler)
   

    ctc_loss = nn.CTCLoss(blank=len(CHAR2ORD)-1, reduction='mean')
    loss_f = nn.MSELoss()
    trainDataset=MyDataset("./phoenix14t.pami0.train.annotations_only.gzip")
    train_loader =DataLoader(trainDataset, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=False,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    trainDataset2=MyDataset2("./phoenix14t.pami0.train.annotations_only.gzip")
    devDataset2=MyDataset2("./phoenix14t.pami0.dev.annotations_only.gzip")
    train_loader2 =DataLoader(trainDataset2, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=False,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    validation_loader =DataLoader(devDataset2, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=False,prefetch_factor=2,collate_fn=custom_collate_fn,shuffle=True)
    val_loss=[]

    for epoch in range(EPOCHS):
        loss_sum=0
        model.train()
        for image,image_label, labels, target_lengths, input_lengths  in train_loader:
            
            image=image.float().cuda().transpose(1,2)
            image_label=image_label.float().cuda().transpose(1,2)
            labels=labels.cuda()
            model.zero_grad()
            # data = data.view(-1, 28*28)
            IMAGE_pred,out=model(image)
            loss_ctc = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)+loss_f(image_label,IMAGE_pred)
            loss_ctc.backward()
            loss_sum+=float(loss_ctc)
            optimizer.step()
            # for p in net.parameters():
            #     p.data.clamp_(-0.5, 0.5)
            #loop.set_postfix(loss = float(ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)),loss_f=float(loss_f(image_label,IMAGE_pred)))
        print('train:',loss_sum/len(train_loader),flush=True)
        valid_epoch_loss = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, 'test.pth')
        if epoch%10==0 and epoch!=0:
            for k in range(10):
                count=0
                model.train()
                loss_sum=0
                valid_epoch_loss = 0
                for image,image_label, labels, target_lengths, input_lengths  in train_loader2:
                
                    image=image.float().cuda().transpose(1,2)
                    image_label=image_label.float().cuda().transpose(1,2)
                    labels=labels.cuda()
                    model.zero_grad()
                    # data = data.view(-1, 28*28)
                    IMAGE_pred,out=model(image)
                    loss_ctc = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
                    loss_ctc.backward()
                    loss_sum+=float(loss_ctc)
                    optimizer.step()
                print('ori_train:',loss_sum/len(train_loader2),flush=True)
                model.eval()
                with torch.no_grad():
                    for image,image_label, labels, target_lengths, input_lengths  in validation_loader:
                        image=image.float().cuda().transpose(1,2)
                        image_label=image_label.float().cuda().transpose(1,2)
                        labels=labels.cuda()
                        IMAGE_pred,out=model(image)
                        loss_ctc = ctc_loss(log_probs=out, targets=labels, target_lengths=target_lengths, input_lengths=input_lengths)
                        valid_epoch_loss+=loss_ctc
                        _, preds = out.max(2)
                        if count<=10:
                            pred1 = preds.transpose(1, 0)[0,:].contiguous().view(-1)

                            print('pred:',decode_out(pred1, ORD2CHAR),' gt:',decode_out(labels[:target_lengths[0]].contiguous().view(-1),ORD2CHAR),flush=True)

                            count+=1
         # 训练时,每个epoch结束后获取验证集loss
        my_lr_scheduler.step(metrics=valid_epoch_loss/len(validation_loader))
        val_loss.append(valid_epoch_loss/len(validation_loader))
        print('Train Epoch: {},Loss: {:.6f}'.format(epoch,loss_sum/len(train_loader)),'valid Epoch: {},Loss: {:.6f}'.format(epoch,valid_epoch_loss/len(validation_loader)),flush=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, 'test.pth')
    print(val_loss)

