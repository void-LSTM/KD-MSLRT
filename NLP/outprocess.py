# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer
from tqdm import tqdm
from torch.optim import lr_scheduler
from WarmupLrScheduler import GradualWarmupScheduler
def decode_out(str_index, characters):
    char_list = []
    for i in range(len(str_index)):
        
        char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list)
if __name__ == "__main__":
    with open('D:\\transformer-main\\word.json') as json_file:
        CHAR2ORD = json.load(json_file)
    BATCH_SIZE=128
    pad_token = '^'
    pad_token_idx = len(CHAR2ORD)
    start_token='*'
    start_token_idx = len(CHAR2ORD)+1
    end_token='$'
    end_token_idx = len(CHAR2ORD)+2
    CHAR2ORD[pad_token] = pad_token_idx
    CHAR2ORD[start_token] = start_token_idx
    CHAR2ORD[end_token] = end_token_idx
    ORD2CHAR = {j:i for i,j in CHAR2ORD.items()}
    out_dim=len(ORD2CHAR)
    trainDataset=TrainDataset("C:\\Users\\wuxin\\Desktop\\KD_test\\xjtlu_out.csv")
    train_loader =DataLoader(trainDataset, batch_size=BATCH_SIZE,
    num_workers=0)#
    

    model = Transformer(out_dim).cuda()
    checkpoint = torch.load('correct_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pred_list=[]
    # loop = tqdm(train_loader, desc='Train')
    for enc_inputs, dec_inputs, dec_outputs in train_loader:  # enc_inputs : [batch_size, src_len]
                                                        # dec_inputs : [batch_size, tgt_len]
                                                        # dec_outputs: [batch_size, tgt_len]

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.long().cuda(), dec_inputs.long().cuda(), dec_outputs.long().cuda()
        outputs, out = model(enc_inputs, dec_inputs)
                                                        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        

            
        input = enc_inputs[0].contiguous().view(-1)
        pred1 = out[0].data.max(1, keepdim=True)[1].contiguous().view(-1)
        gt=dec_outputs[0].contiguous().view(-1)
        pred_list.append([decode_out(input, ORD2CHAR),decode_out(pred1, ORD2CHAR),decode_out(gt,ORD2CHAR)])
        print(' ori:',decode_out(input, ORD2CHAR))
        print(' pred:',decode_out(pred1, ORD2CHAR))
        print(' gt:',decode_out(gt,ORD2CHAR))
        print('______________________________________________________________')


