# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer
from tqdm import tqdm
from torch.optim import lr_scheduler
from WarmupLrScheduler import GradualWarmupScheduler
def decode_out(str_index, characters,pad_token_idx,end_token_idx):
    char_list = []
    
    for i in range(len(str_index)):
        if str_index[i].item()==pad_token_idx or str_index[i].item()==end_token_idx:
            break
        
        char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list)
dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
gloss_dict = np.load(dict_path, allow_pickle=True).item()
CHAR2ORD= dict((v[0], k) for k, v in gloss_dict.items())   
pad_token = '^'
pad_token_idx = 0
start_token='*'
start_token_idx = len(CHAR2ORD)+1
end_token='$'
end_token_idx = len(CHAR2ORD)+2
CHAR2ORD[pad_token] = pad_token_idx
CHAR2ORD[start_token] = start_token_idx
CHAR2ORD[end_token] = end_token_idx
if __name__ == "__main__":
    ORD2CHAR = {i:j for i,j in CHAR2ORD.items()}
    BATCH_SIZE=256

    trainDataset=MyDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\NLP\\PHOENIX_parquet.csv")
    train_loader =DataLoader(trainDataset, batch_size=BATCH_SIZE,
    num_workers=1, pin_memory=True,prefetch_factor=2,shuffle=True)
    valDataset=TrainDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\xjtlu_out.csv")
    Val_loader =DataLoader(valDataset, batch_size=BATCH_SIZE,
    num_workers=1, pin_memory=True,prefetch_factor=2,shuffle=True)
    
    
    model = Transformer(len(CHAR2ORD)).cuda()
    criterion = nn.CrossEntropyLoss()         # 忽略 占位符 索引为0.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # checkpoint = torch.load('correct_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    model2 = Transformer(len(CHAR2ORD)).cuda()
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)
    # exp_lr_scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.8,patience=5,verbose=False)
    # my_lr_scheduler=GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=200, after_scheduler=exp_lr_scheduler)
    # checkpoint = torch.load('correct_model2.pth')
    # model2.load_state_dict(checkpoint['model_state_dict'])
    for epoch in range(10000):
        model.train()
        count=0
        losssum=0
        loop = tqdm(train_loader, desc='Train')
        for enc_inputs, dec_inputs, dec_outputs in loop:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.long().cuda(), dec_inputs.long().cuda(), dec_outputs.long().cuda()
            outputs, out = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losssum+=float(loss)
            enc_inputs2=out.detach().data.max(2, keepdim=True)[1].reshape(enc_inputs.shape[0],enc_inputs.shape[1])
            outputs2, out2 = model2(enc_inputs2, dec_inputs)
            loss2 = criterion(outputs2, dec_outputs.view(-1))
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
           
            loop.set_postfix(loss=float(loss),loss2=float(loss2))
            
        # if count%20==0:
            
        #     input = enc_inputs[0].contiguous().view(-1)
            
        #     pred1 = out2[0].data.max(1, keepdim=True)[1].contiguous().view(-1)
        #     gt=dec_outputs[0].contiguous().view(-1)

        #     print(' ori:',decode_out(input, ORD2CHAR,pad_token_idx,end_token_idx))
        #     print(' pred:',decode_out(pred1, ORD2CHAR,pad_token_idx,end_token_idx))
        #     print(' gt:',decode_out(gt,ORD2CHAR,pad_token_idx,end_token_idx))
        #         # print('loss:',losssum/count)
                
        count+=1
        model.eval()
        lossval=0
        lossreal=0
        for enc_inputs, dec_inputs, dec_outputs in Val_loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.long().cuda(), dec_inputs.long().cuda(), dec_outputs.long().cuda()
            outputs, out = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            enc_inputs2=out.detach().data.max(2, keepdim=True)[1].reshape(enc_inputs.shape[0],enc_inputs.shape[1])
            outputs2, out2 = model2(enc_inputs2, dec_inputs)
            loss = criterion(outputs2, dec_outputs.view(-1))

            lossval+=float(loss)
            input = enc_inputs[0].contiguous().view(-1)
            
            pred1 = out2[0].data.max(1, keepdim=True)[1].contiguous().view(-1)
            gt=dec_outputs[0].contiguous().view(-1)

            print(' ori:',decode_out(input, ORD2CHAR,pad_token_idx,end_token_idx))
            print(' pred:',decode_out(pred1, ORD2CHAR,pad_token_idx,end_token_idx))
            print(' gt:',decode_out(gt,ORD2CHAR,pad_token_idx,end_token_idx))
        # # my_lr_scheduler.step(lossval/len(Val_loader))
        print(epoch,losssum/len(train_loader),lossval/len(Val_loader))
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                }, 'correct_model.pth')
        # torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model2.state_dict(),
        #         }, 'correct_model2.pth')

        print("保存模型")
