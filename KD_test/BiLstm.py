import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

d_model = 512   # 字 Embedding 的维度
n_heads = 1     # Multi-Head Attention设置为8

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor(in_dim, out_dim))

        nn.init.xavier_uniform_(self.weight1, gain=nn.init.calculate_gain('relu'))


    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight1, dim=0))

        return outputs

class down(nn.Module):
    def __init__(self,out_dim):
        super(down, self).__init__()
        self.conv3d= nn.Sequential(
            nn.Conv3d(3, 1, kernel_size=1, stride=(1,1,1), bias=False),
        )

class BiLSTM_SA(nn.Module):

    def __init__(self, config,tgt_vocab_size):
        super().__init__()
        self.LINEAR = nn.Sequential(
            nn.Conv1d(276, 512, 5, 1, 0),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, 1, 0),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, config.dense_hidden_size, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )
        self.LSTM = nn.LSTM(config.embed_size, config.lstm_hidden_size,
                            num_layers=config.num_layers, batch_first=True,dropout=0,
                            bidirectional=True)
        self.norm_LSTM=nn.LayerNorm(config.lstm_hidden_size * 2)
        self.ffn = nn.Sequential(
            nn.Linear(config.lstm_hidden_size * 2,config.dense_hidden_size),
            nn.ReLU()
                            )
        self.norm_ffn=nn.LayerNorm(config.dense_hidden_size)
        # self.decoder_linear_FRAME=nn.Linear(54,475)
        # self.decoder_linear=nn.Linear(2048,276)
        self.classifier = NormLinear(config.dense_hidden_size,
                                    tgt_vocab_size)
        # self.classifier = nn.se
        # NormLinear(config.dense_hidden_size,
        #                             tgt_vocab_size)


    def forward(self, inputs):   

        inputs=inputs.transpose(1,2)
        enc_inputs=self.LINEAR(inputs)
        enc_inputs=enc_inputs.transpose(1,2)

        conv_pred=self.classifier(self.norm_ffn(enc_inputs)).transpose(0,1)
        # conv_pred=F.log_softmax(conv_pred, dim=2)


        lstm_hidden_states, _ = self.LSTM(enc_inputs)

        lstm_hidden_states=self.norm_LSTM(lstm_hidden_states)


        ffn_outputs =self.ffn(lstm_hidden_states)
        ffn_outputs=self.norm_ffn(ffn_outputs)
        # de_out=self.decoder_linear_FRAME(ffn_outputs.transpose(1,2))
        # de_out=self.decoder_linear(de_out.transpose(1,2))
        logits = self.classifier(ffn_outputs)
        logits=logits.transpose(0,1)
        # logits=F.log_softmax(logits, dim=2)
        de_out=0


        
        return logits,de_out,conv_pred



class BiLSTM_SA_temp(nn.Module):

    def __init__(self, config,tgt_vocab_size):
        super().__init__()
        self.LINEAR = nn.Sequential(
            nn.Conv1d(276, 512, 5, 1, 0),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, 1, 0),
            nn.ReLU(),
            nn.Conv1d(512, 512, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(512, config.dense_hidden_size, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )
        self.LSTM = nn.LSTM(config.embed_size, config.lstm_hidden_size,
                            num_layers=config.num_layers, batch_first=True,dropout=0,
                            bidirectional=True)
        self.norm_LSTM=nn.LayerNorm(config.lstm_hidden_size * 2)
        self.ffn = nn.Sequential(
            nn.Linear(config.lstm_hidden_size * 2,config.dense_hidden_size),
            nn.ReLU(),
                            )
        self.norm_ffn=nn.LayerNorm(config.dense_hidden_size)
        # self.decoder_linear_FRAME=nn.Linear(54,475)
        # self.decoder_linear=nn.Linear(2048,276)
        self.classifier = NormLinear(config.dense_hidden_size,
                                    tgt_vocab_size)
        # self.classifier = nn.se
        # NormLinear(config.dense_hidden_size,
        #                             tgt_vocab_size)


    def forward(self, inputs):   

        
        inputs=inputs.transpose(1,2)
        enc_inputs=self.LINEAR(inputs)
        enc_inputs=enc_inputs.transpose(1,2)

        conv_pred=self.classifier(self.norm_ffn(enc_inputs)).transpose(0,1)
        # conv_pred=F.log_softmax(conv_pred, dim=2)


        lstm_hidden_states, _ = self.LSTM(enc_inputs)

        lstm_hidden_states=self.norm_LSTM(lstm_hidden_states)


        ffn_outputs =self.ffn(lstm_hidden_states)
        ffn_outputs=self.norm_ffn(ffn_outputs)
        # de_out=self.decoder_linear_FRAME(ffn_outputs.transpose(1,2))
        # de_out=self.decoder_linear(de_out.transpose(1,2))
        logits = self.classifier(ffn_outputs)
        logits=logits.transpose(0,1)
        # logits=F.log_softmax(logits, dim=2)
        de_out=0


        
        return logits