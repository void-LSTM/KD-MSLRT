import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import numpy as np
from Dataset import *
from tqdm import tqdm
from zhengliu import SLRModel
from ctcdecoder import*
def find_duplicates(lst):
    return list(set([x for x in lst if lst.count(x) > 1]))
def decode_out(str_index, characters):
    char_list = []
    print(str_index)
    for i in range(len(str_index)):
        if str_index[i] != 0 :
            char_list.append(characters[str_index[i].item()])
    return ' '.join(char_list)
def custom_collate_fn(batch, T=30):
        items = list(zip(*batch))
        items[0] = default_collate(items[0])
        items[1] = default_collate(items[1])
        labels = list(items[2])
        items[2] = []
        items[3] = default_collate(items[3])
        items[4] = default_collate(items[4])
        target_lengths = torch.zeros((len(batch,)), dtype=torch.int)
        input_lengths = torch.zeros(len(batch,), dtype=torch.int)
        for idx, label in enumerate(labels):
            # 记录每个图片对应的字符总数
            target_lengths[idx] = len(label)
            # 将batch内的label拼成一个list
            items[2].extend(label)
            # input_lengths 恒为 T
            input_lengths[idx] = T
        return items[0],items[1], torch.tensor(items[2]), target_lengths, input_lengths,items[3], items[4]
if __name__ == '__main__':
    dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\feature_map\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
    gloss_dict = np.load(dict_path, allow_pickle=True).item()
    gloss_dict= dict((v[0], k) for k, v in gloss_dict.items())
    conv_logits={}
    sequence_logits={}
    loadData = np.load(dict_path,allow_pickle=True)
    test_dict=loadData.item()
    trainDataset2=MyDataset2("D:\\phoenix\\phoenix-2014.v3.tar\\phoenix2014-release\\phoenix-2014-multisigner\\annotations\\manual\\train.corpus.csv")
    devDataset2=MyDataset2("D:\\phoenix\\phoenix-2014.v3.tar\\phoenix2014-release\\phoenix-2014-multisigner\\annotations\\manual\\dev.corpus.csv")
    train_loader2 =DataLoader(trainDataset2, batch_size=1,
    num_workers=2,collate_fn=custom_collate_fn,shuffle=True,pin_memory=True)
    # dev_loader2 =DataLoader(devDataset2, batch_size=1,
    # num_workers=2, pin_memory=False,collate_fn=custom_collate_fn,shuffle=True)
    loop=tqdm(train_loader2,desc='train')
    net=SLRModel(len(gloss_dict)+1,'resnet18',2).cuda()
    checkpoint=torch.load('C:\\Users\\wuxin\\Desktop\\KD_test\\feature_map\\CorrNet\\dev_18.90_PHOENIX14.pt')
    net.load_state_dict(checkpoint["model_state_dict"])
    temp=[]
    for image,image_label, labels, target_lengths, input_lengths,path,vedio_len  in loop:
        with torch.no_grad():
            image=image.cuda()
            lenin=torch.tensor(vedio_len[0])
            out=net(image,lenin)
            conv_logits[path[0]]=out['conv_logits'].cpu().numpy()
            sequence_logits[path[0]]=out['sequence_logits'].cpu().numpy()

            # pred, score, labels_p = ctc_beam_search_decode(out['sequence_logits'].softmax(-1).permute(1, 0, 2)[0].cpu().numpy() , beam_size=10, blank=0)

            # pred1 = preds.transpose(1, 0)[0,:].contiguous().view(-1)
            # print(labels)

            # print('pred:',decode_out(pred[0], gloss_dict),' gt:',decode_out(labels[:target_lengths[0]].contiguous().view(-1),gloss_dict))
 
    np.save('conv_logits.npy', conv_logits)
    np.save('sequence_logits.npy', sequence_logits)
    load_data = np.load('sequence_logits.npy', allow_pickle=True)
    load_data=load_data.item()
    print(len(load_data))

