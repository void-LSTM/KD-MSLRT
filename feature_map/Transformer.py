import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
  # 字 Embedding 的维度
d_ff = 1024     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 1   # 有多少个encoder和decoder
n_heads = 6     # Multi-Head Attention设置为8


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])           # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])           # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table)       # enc_inputs: [seq_len, d_model]

    def forward(self, enc_inputs):                                  # enc_inputs: [batch_size, seq_len, d_model]
        device=enc_inputs.device
        self.pos_table = self.pos_table.to(device)  
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)


def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)            # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):                              # Q: [batch_size, n_heads, len_q, d_k]
                                                                        # K: [batch_size, n_heads, len_k, d_k]
                                                                        # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)    # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                 # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.LayerNorm=nn.LayerNorm(d_model)
    def forward(self, input_Q, input_K, input_V):    # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)    # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)       # V: [batch_size, n_heads, len_v(=len_k), d_v]                               
        context, attn = ScaledDotProductAttention()(Q, K, V)             # context: [batch_size, n_heads, len_q, d_v]
                                                                                    # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)                    # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                   # [batch_size, len_q, d_model]
        return self.LayerNorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))
        self.LayerNorm=nn.LayerNorm(d_model)

    def forward(self, inputs):                                  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.LayerNorm(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self,d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model)                   # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(d_model)                      # 前馈神经网络

    def forward(self, enc_inputs):              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                                    # enc_outputs: [batch_size, src_len, d_model],
                                               )  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
# class NormLinear(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(NormLinear, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
#         nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

#     def forward(self, x):
#         outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
#         return outputs
# class BiLstm(nn.Module):
#     def __init__(self,d_model,tgt_vocab_size,lstm_hidden_size=512,num_layers=1,dense_hidden_size=512):
#         super(BiLstm, self).__init__()
#         self.LSTM = nn.LSTM(d_model, lstm_hidden_size,
#                             num_layers=num_layers, batch_first=True,dropout=0,
#                             bidirectional=True)
#         self.norm_LSTM=nn.LayerNorm(lstm_hidden_size * 2)
#         self.ffn = nn.Sequential(
#             nn.Linear(lstm_hidden_size * 2,dense_hidden_size),
#             nn.ReLU(),
#             # nn.Dropout(0.3)
#                             )
#         self.norm_ffn=nn.LayerNorm(dense_hidden_size)
#         self.classifier = NormLinear(dense_hidden_size,
#                                     tgt_vocab_size)
#     def forward(self, inputs):
#         lstm_hidden_states, _ = self.LSTM(inputs)
#         lstm_hidden_states=self.norm_LSTM(lstm_hidden_states)
#         ffn_outputs =self.ffn(lstm_hidden_states)
#         ffn_outputs=self.norm_ffn(ffn_outputs)
#         logits = self.classifier(ffn_outputs)
#         logits=logits.transpose(0,1)
#         logits=F.log_softmax(logits, dim=2)
#         return logits

# class Encoder(nn.Module):
#     def __init__(self,tgt_vocab_size,d_model):
#         super(Encoder, self).__init__()
#         self.LINEAR = nn.Sequential(
#             nn.Conv1d(1000, 512, 7, 1, 0),
#             nn.Sigmoid(),
#             nn.BatchNorm1d(512),
#             nn.Conv1d(512, 512, 7, 1, 0),
#             nn.Tanh(),
#             nn.BatchNorm1d(512),
#             nn.Conv1d(512, 512, 7, 2, 0),
#             nn.Tanh(),
#             nn.BatchNorm1d(512),
#             nn.Conv1d(512, 512, 7, 2, 0),
#             nn.Tanh(),
#             nn.BatchNorm1d(512),

#         )
#         self.projection = BiLstm(d_model, tgt_vocab_size)

#     def forward(self, enc_inputs):    
#         enc_inputs=self.LINEAR(enc_inputs)                                   # enc_inputs: [batch_size, src_len]
#         enc_inputs=enc_inputs.transpose(1, 2)
#         pred=self.projection(enc_inputs)
#         return enc_inputs, pred
