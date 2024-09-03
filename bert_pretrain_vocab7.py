'''
Everything is uncanonized
all_smiles3 database is used
all_smiles3_vocab is used
'''


import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from build_vocab import WordVocab
# from dataset1_orig import Seq2seqDataset
import dataset2
# import dataset5

import copy
from typing import Optional, Any

import torch
from torch import Tensor
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import resnet18
import random


seed = 101
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
random.seed(seed)

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4
SEP = 5

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Embedding_bert(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments):
        super(Embedding_bert, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).cuda()  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model).cuda()  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model).cuda()  # segment(token type) embedding
        # self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).cuda()
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        # embedding = self.tok_embed(x) + self.seg_embed(seg)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        # return self.norm(embedding)
        return embedding

class BERT1(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments, n_layers, d_k, d_v, n_heads, d_ff, num_classes):
        super(BERT1, self).__init__()
        # BERT1(len(vocab), args.hidden, maxlen, n_segments, args.n_layer, d_k, d_v, args.n_head, d_ff, args.num_classes).cuda()
        # self.embed = nn.Embedding(vocab_size, d_model)
        self.embed = Embedding_bert(vocab_size, d_model, maxlen, n_segments)
        # self.pe = PositionalEncoding(d_model, dropout=0)
        # self.embedding = Embedding(vocab_size, d_model, maxlen, n_segments)
        # self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                        dim_feedforward=d_ff, dropout=0)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=0) for _ in range(n_layers)])
        # self.fc = nn.Linear(d_model, d_model)
        # self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)

        # self.classifier1 = nn.Linear(d_model, 512)
        # self.classifier2 = nn.Linear(512, 256)
        # self.classifier3 = nn.Linear(256, num_classes)
        self.resnet18 = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock,
                                        num_classes=num_classes)
        self.out3 = nn.Linear(in_features=512, out_features=num_classes)
        # decoder is shared with embedding layer
        embed_weight = self.embed.tok_embed.weight
        n_vocab = vocab_size
        n_dim = d_model
        # n_vocab, n_dim = embed_weight.size()

        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        # embedded = self.embed(input_ids)
        embedded = self.embed(input_ids, segment_ids)
        # embedded = self.pe(embedded)

        # output = self.embedding(input_ids, segment_ids)
        hidden = self.encoder(embedded)
        hidden = hidden.permute(1,0,2)
        # print(hidden.shape)
        hidden1 = hidden.reshape(hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])
        # enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # print(enc_self_attn_mask.shape)
        # for layer in self.layers:
        #     output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output = layer(output)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        # h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        # logits_clsf1 = self.classifier1(h_pooled) # [batch_size, 2]
        # logits_clsf2 = self.classifier2(logits_clsf1)
        # logits_clsf3 = self.classifier3(logits_clsf2)

        # output = output.reshape(output.shape[0], 1, output.shape[1], output.shape[2])
        out = self.resnet18(hidden1)
        logits_clsf = self.out3(out)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, hidden.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(hidden, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_clsf, logits_lm


class TrfmSeq2seq(nn.Module):

    # def append_dropout(model, rate=0.1):
    #     for name, module in model.named_children():
    #         if len(list(module.children())) > 0:
    #             TrfmSeq2seq.append_dropout(module)
    #         if isinstance(module, nn.ReLU):
    #             # new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
    #             new = nn.Sequential(module, nn.Dropout(p=rate))
    #             setattr(model, name, new)

    # model = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock, num_classes=56)
    #
    # append_dropout(model)
    # print(model)

    def __init__(self, in_size, hidden_size, num_classes, out_size, n_layers, n_head, dropout=0.0):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.n_head = n_head
        self.embed = nn.Embedding(self.in_size, self.hidden_size)
        self.pe = PositionalEncoding(self.hidden_size, self.dropout)
        # self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_head, dim_feedforward=self.hidden_size, dropout=self.dropout)
        #self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size)
        #self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size)
        #self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)
        self.resnet18 = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock, num_classes=self.num_classes)
        # self.append_dropout(self.resnet18)
        # self.conv1 = nn.
        # self.pooler = nn.AvgPool1d(kernel_size=500)

        # self.out = nn.Linear(hidden_size, out_size)
        # self.out2 = nn.Linear(512, 256)
        self.out3 = nn.Linear(in_features=512, out_features=num_classes)



    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded)  # (T,B,H)
        hidden = self.encoder(embedded)
        hidden = hidden.permute(1,0,2)
        # print(hidden.shape)
        hidden = hidden.reshape(hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])
        out = self.resnet18(hidden)
        # print(out.shape)
        # embedded = torch.flatten(embedded)
        # hidden = hidden.max(dim=1)[0]
        # hidden = self.pooler(hidden).squeeze()
        # out = self.out2(out)
        out = self.out3(out)
        # out = self

        # hidden = self.trfm(embedded, embedded)  # (T,B,H)
        # out = self.out(hidden)  # (T,B,V)
        # out2 = self.out2(hidden)

        # out = F.log_softmax(out, dim=2)  # (T,B,V)
        return out  # (T,B,V)

    # def _encode(self, src):
    #     # src: (T,B)
    #     embedded = self.embed(src)  # (T,B,H)
    #     embedded = self.pe(embedded)  # (T,B,H)
    #     output = embedded
    #     for i in range(self.trfm.encoder.num_layers - 1):
    #         output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
    #     penul = output.detach().numpy()
    #     output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
    #     if self.trfm.encoder.norm:
    #         output = self.trfm.encoder.norm(output)  # (T,B,H)
    #     output = output.detach().numpy()
    #     # mean, max, first*2
    #     return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0, :, :], penul[0, :, :]])  # (B,4H)
    #
    # def encode(self, src):
    #     # src: (T,B)
    #     batch_size = src.shape[1]
    #     if batch_size <= 100:
    #         return self._encode(src)
    #     else:  # Batch is too large to load
    #         print('There are {:d} molecules. It will take a little time.'.format(batch_size))
    #         st, ed = 0, 100
    #         out = self._encode(src[:, st:ed])  # (B,4H)
    #         while ed < batch_size:
    #             st += 100
    #             ed += 100
    #             out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
    #         return out

# class TransformerEncoderLayer(Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#     """
#
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer, self).__init__()
#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model)
#
#         self.norm1 = LayerNorm(d_model)
#         self.norm2 = LayerNorm(d_model)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer, self).__setstate__(state)
#
#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



def parse_arguments():
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_epoch', '-e', type=int, default=800, help='number of epochs')
    parser.add_argument('--num_classes', '-c', type=int, default=65, help='number of classes')
    # parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_db1_drugs_orig.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_all_smiles3.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--data', '-d', type=str, default='./data/DB1_data_allFolds', help='train corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='./result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='Pretrain_Bert', help='model name')
    parser.add_argument('--seq_len', type=int, default=500, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')  # default=16
    parser.add_argument('--hidden', type=int, default=256, help='length of hidden vector')
    parser.add_argument('--n_layer', '-l', type=int, default=6, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--lr', type=float, default=1e-5, help='Adam learning rate')
    parser.add_argument('--gpu', metavar='N', type=int, nargs='+', help='list of GPU IDs to use')
    return parser.parse_args()


r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
# from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

#
# def default_convert(data):
#     r"""Converts each NumPy array data field into a tensor"""
#     elem_type = type(data)
#     if isinstance(data, torch.Tensor):
#         return data
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         # array of string classes and object
#         if elem_type.__name__ == 'ndarray' \
#                 and np_str_obj_array_pattern.search(data.dtype.str) is not None:
#             return data
#         return torch.as_tensor(data)
#     elif isinstance(data, container_abcs.Mapping):
#         return {key: default_convert(data[key]) for key in data}
#     elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
#         return elem_type(*(default_convert(d) for d in data))
#     elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
#         return [default_convert(d) for d in data]
#     else:
#         return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


# def collate_temp(batch):
#     r"""Puts each data field into a tensor with outer dimension batch size"""
#
#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         # print("Starting Debug: ")
#         # for i in range(len(batch)):
#         #     print(len(batch[i]))
#
#         return torch.stack(batch, 0, out=out)
#         # return batch
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(default_collate_err_msg_format.format(elem.dtype))
#
#             return collate_temp([torch.as_tensor(b) for b in batch])
#         elif elem.shape == ():  # scalars
#             return torch.as_tensor(batch)
#     elif isinstance(elem, float):
#         return torch.tensor(batch, dtype=torch.float64)
#     elif isinstance(elem, int_classes):
#         return torch.tensor(batch)
#     elif isinstance(elem, string_classes):
#         return batch
#     elif isinstance(elem, container_abcs.Mapping):
#         return {key: collate_temp([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
#         return elem_type(*(collate_temp(samples) for samples in zip(*batch)))
#     elif isinstance(elem, container_abcs.Sequence):
#         # check to make sure that the elements in batch have consistent size
#         it = iter(batch)
#         elem_size = len(next(it))
#         if not all(len(elem) == elem_size for elem in it):
#             raise RuntimeError('each element in list of batch should be of equal size')
#         transposed = zip(*batch)
#         return [collate_temp(samples) for samples in transposed]
#
#     raise TypeError(default_collate_err_msg_format.format(elem_type))

def evaluate_bert(model, test_loader):
    model.eval()
    total_loss = 0
    acc = 0
    # targets_list = []
    # outputs_list = []
    criterion = nn.CrossEntropyLoss()
    for b, d in enumerate(test_loader):
        input_ids = d[0].cuda()
        segment_ids = d[1].cuda()
        masked_pos = d[2].cuda()
        masked_tokens = d[3].cuda()
        target = d[4].cuda()

        with torch.no_grad():
            logits_clsf, logits_lm = model(torch.t(input_ids), torch.t(segment_ids), masked_pos)

        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        # loss_clsf = criterion_bert(logits_clsf, target)  # for sentence classification
        loss = loss_lm
        # logits_clsf = model(input_ids)

        # loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        # loss_lm = (loss_lm.float()).mean()
        # loss_clsf = criterion(logits_clsf, target)  # for sentence classification
        # loss = loss_clsf

        total_loss += loss.item()


        # pred = torch.max(logits_clsf, axis=1)[1]
        # acc += torch.sum(pred == target).item()


    final_loss = total_loss / len(test_loader)
    # return final_loss, acc/len(test_loader.dataset)
    return final_loss




def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    acc = 0
    # targets_list = []
    # outputs_list = []
    criterion = nn.CrossEntropyLoss()
    for b, d in enumerate(test_loader):
        sm = d[0]
        target = d[2]
        sm = torch.t(sm.cuda())  # (T,B)
        target = target.cuda()
        with torch.no_grad():
            output = model(sm)  # (T,B,V)

        loss = criterion(output, target)
        total_loss += loss.item()
        pred = torch.max(output, axis=1)[1]
        acc += torch.sum(pred == target).item()
        # targets_list.append(sm.detach().cpu().numpy())
        # outputs_list.append(output.detach().cpu().numpy())

    # data = {}
    # data['eval_targets'] = targets_list
    # data['eval_outputs'] = outputs_list
    final_loss = total_loss / len(test_loader)
    return final_loss, acc/len(test_loader.dataset)

train_loss_list_trfm = []
eval_loss_list_trfm = []
s1_loss_list_trfm = []
s2_loss_list_trfm = []
train_acc_list_trfm = []
val_acc_list_trfm = []
s1_acc_list_trfm = []
s2_acc_list_trfm = []

train_loss_list_bert = []
eval_loss_list_bert = []
s1_loss_list_bert = []
s2_loss_list_bert = []
train_acc_list_bert = []
val_acc_list_bert = []
s1_acc_list_bert = []
s2_acc_list_bert = []

# resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

#     tensor = torch.rand([1, 1, 500, 224])
#     model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1000)
# model_sub = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock, num_classes=1000)

def main():

    # eval_data = []
    args = parse_arguments()
    assert torch.cuda.is_available()
    # args.batch_size = 1

    print('Loading dataset...')
    with open('./data/DB1_data_allFolds', 'rb') as f:
        a = pickle.load(f)

    train_fold, valid_fold, s1_fold, s2_fold = a[0:4]
    train_data = train_fold[0]
    valid_data = valid_fold[0]
    s1_data = s1_fold[0]
    s2_data = s2_fold[0]


    max_pred = 65
    # Building pretraining Dataset:


    random.seed(101)
    pretrain_df = pd.read_csv('./data/all_smiles3.csv')
    smiles = list(pretrain_df['canonical_smiles'])
    pretrain_data = []
    # ['CN(C)CCOC(=O)C(C1=CC=CC=C1)C1(O)CCCC1', 1, 'CCN(CC)C(C)CN1C2=CC=CC=C2SC2=CC=CC=C12']
    for i in range(len(smiles)):
        pretrain_data.append([smiles[i], 0, smiles[random.randint(0,len(smiles)-1)]])


    vocab = WordVocab.load_vocab(args.vocab)
    # dataset_train = Seq2seqDataset(train_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_valid = Seq2seqDataset(valid_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_s1 = Seq2seqDataset(s1_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_s2 = Seq2seqDataset(s2_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_test = Seq2seqDataset(train_data[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_test1 = dataset2.Seq2seqDataset(train_data[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_pretrain = dataset2.Seq2seqDataset(pretrain_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)

    test_size = 1000  # 10000
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset_pretrain,[len(dataset_pretrain) - test_size, test_size])

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    # s1_loader = DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    # s2_loader = DataLoader(dataset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    # pretrain_loader = DataLoader(dataset_pretrain, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)

    # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=args.n_worker)
    # loader_test1 = DataLoader(dataset_test1, batch_size=1, shuffle=True, num_workers=args.n_worker)

    print('Train size:', len(dataset_train))
    print('Test size:', len(dataset_valid))
    # print('s1 size:', len(dataset_s1))
    # print('s2 size:', len(dataset_s2))
    # print('pretrain size:', len(dataset_pretrain))

    # del dataset, train, test
    # torch.manual_seed(101)
    # torch.cuda.manual_seed(101)
    # torch.cuda.manual_seed_all(101)
    # model = TrfmSeq2seq(len(vocab), args.hidden, args.num_classes, len(vocab), args.n_layer, args.n_head).cuda()
    # print(model)

    maxlen = 500
    n_segments = 2
    d_k = d_v = 64  # dimension of K(=Q), V
    d_ff = args.hidden
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    model_bert = BERT1(len(vocab), args.hidden, maxlen, n_segments, args.n_layer, d_k, d_v, args.n_head, d_ff, args.num_classes).cuda()


    optimizer_bert = optim.Adam(model_bert.parameters(), lr=args.lr) # Add weight decay weight_decay=1e-5 for L2 regularization
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # Add weight decay weight_decay=1e-5 for L2 regularization
    criterion_bert = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # print(model)
    # print('Total parameters:', sum(p.numel() for p in model.parameters()))



    best_loss = 1000
    best_epoch = 0
    training_loss_list = []
    for e in range(1, args.n_epoch):
        print(">>> Epoch:  ", e)
        for b, d in tqdm(enumerate(train_loader)):
            # break
            # a = 50
            input_ids = d[0].cuda()
            segment_ids = d[1].cuda()
            masked_pos = d[2].cuda()
            masked_tokens = d[3].cuda()
            target = d[4].cuda()
            # break
            # # Training TRFM model:
            # # target = target.cuda()
            # optimizer.zero_grad()
            # output = model(torch.t(sm))  # (T,B,V)
            # # loss = F.nll_loss(output.view(-1, len(vocab)), sm.contiguous().view(-1), ignore_index=PAD)
            # loss = criterion(output, target)
            #
            # # loss = F.multi(output, target)
            # loss.backward()
            # optimizer.step()
            # if b % 100 == 0:
            #     print('TRFM: Train {:3d}: iter {:5d} | loss {}'.format(e, b, loss.item()))
            # # if b % 100 == 0:


            # Training Bert model:
            optimizer_bert.zero_grad()
            # logits_clsf = model_bert(torch.t(sm), torch.t(segment_ids))
            logits_clsf, logits_lm = model_bert(torch.t(input_ids), torch.t(segment_ids), masked_pos)
            loss_lm = criterion_bert(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
            loss_lm = (loss_lm.float()).mean()
            # loss_clsf = criterion_bert(logits_clsf, target)  # for sentence classification
            loss = loss_lm
            # loss = F.multi(output, target)
            loss.backward()
            optimizer_bert.step()
            if b % 100 == 0:
                print('BERT1: Train {:3d}: iter {:5d} | loss {}'.format(e, b, loss.item()))

        # # Evaluting TRFM model every epoch
        # loss_train, acc_train = evaluate(model, train_loader)
        # train_loss_list_trfm.append(loss_train)
        # train_acc_list_trfm.append(acc_train)
        # # eval_data.append(data)
        # print('TRFM: Train {:3d}: iter {:5d} | loss {} | acc {}'.format(e, b, loss_train, acc_train))
        # loss_val, acc_val = evaluate(model, test_loader)
        # eval_loss_list_trfm.append(loss_val)
        # val_acc_list_trfm.append(acc_val)
        #
        # print('TRFM: Val {:3d}: iter {:5d} | loss {} | acc {}'.format(e, b, loss_val, acc_val))
        # loss_s1, acc_s1 = evaluate(model, s1_loader)
        # s1_loss_list_trfm.append(loss_s1)
        # s1_acc_list_trfm.append(acc_s1)
        #
        # print('TRFM: s1 {:3d}: iter {:5d} | loss {} | acc{}'.format(e, b, loss_s1, acc_s1))
        # loss_s2, acc_s2 = evaluate(model, s2_loader)
        # s2_loss_list_trfm.append(loss_s2)
        # s2_acc_list_trfm.append(acc_s2)
        #
        # print('TRFM: s2 {:3d}: iter {:5d} | loss {} | acc{}'.format(e, b, loss_s2, acc_s2))
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_loss or loss < best_loss:
        #     print("[!] saving model...")
        #     # outputs_all.append(output.detach().cpu().numpy())
        #     # targets_all.append(sm.detach().cpu().numpy())
        #
        #     output_save = output.detach().cpu().numpy()
        #     targets_save = sm.detach().cpu().numpy()
        #     train_data_path = './result/' + 'trainData_' + str(e) + '_' + str(b) + '.pkl'
        #     # targets_path = './result/' + 'train_targets_' + str(e) + '_' + str(b) + '.csv'
        #     eval_path = './result/' + 'eval_data_' + str(e) + '_' + str(b) + '.pkl'
        #     # data_df = pd.DataFrame(data)
        #     train_data = {}
        #     train_data['train_targets'] = targets_save
        #     train_data['output_save'] = output_save
        #     # with open(train_data_path, "wb") as output_file:
        #     #     pickle.dump(train_data, output_file)
        #
        #     # with open(eval_path, "wb") as output_file:
        #     #     pickle.dump(data, output_file)
        #     # train_df = pd.DataFrame(train_data)
        #     # train_df.to_csv(train_data_path, index=False)
        #     # data_df.to_csv(eval_path, index=False)
        #     if not os.path.isdir("model"):
        #         os.makedirs("model")
        #     best_epoch = e
        #     # torch.save(model.state_dict(), './model/trfm_new_%d_%d.pkl' % (e, b))
        #     best_loss = loss

        # # Evaluating loss for BERT model:
        train_loss = evaluate_bert(model_bert, train_loader)
        training_loss_list.append(train_loss)
        print("Train Loss:   Epoch: ", e, ' || loss: ', train_loss)
        test_loss = evaluate_bert(model_bert, test_loader)
        print("Test Loss:   Epoch: ", e, ' || loss: ', test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = e
            print("Saving model")
            torch.save(model_bert.state_dict(), './model/bert_pretrain_vocab7_%d_%d.pkl' % (e, b))

        # model_bert.load_state_dict(torch.load("./model/bert_pretrain_vocab3_588_2372.pkl"))

        # train_loss_list_bert.append(loss_train)
        # train_acc_list_bert.append(acc_train)
        # # eval_data.append(data)
        # print('BERT: Train {:3d}: iter {:5d} | loss {} | acc {}'.format(e, b, loss_train, acc_train))
        # loss_val, acc_val = evaluate_bert(model_bert, test_loader)
        # eval_loss_list_bert.append(loss_val)
        # val_acc_list_bert.append(acc_val)
        #
        # print('BERT: Val {:3d}: iter {:5d} | loss {} | acc {}'.format(e, b, loss_val, acc_val))
        # loss_s1, acc_s1 = evaluate_bert(model_bert, s1_loader)
        # s1_loss_list_bert.append(loss_s1)
        # s1_acc_list_bert.append(acc_s1)
        #
        # print('BERT: s1 {:3d}: iter {:5d} | loss {} | acc{}'.format(e, b, loss_s1, acc_s1))
        # loss_s2, acc_s2 = evaluate_bert(model_bert, s2_loader)
        # s2_loss_list_bert.append(loss_s2)
        # s2_acc_list_bert.append(acc_s2)
        #
        # print('BERT: s2 {:3d}: iter {:5d} | loss {} | acc{}'.format(e, b, loss_s2, acc_s2))
        #


    # with open(r"./result/complete_trfm10.eval.pkl", "wb") as output_file:
    #     pickle.dump([train_acc_list, train_loss_list, val_acc_list, eval_loss_list, s1_acc_list, s1_loss_list, s2_acc_list, s2_loss_list], output_file)

    # print("TRFM: The Best Val accuracy: ", max(val_acc_list_trfm), " | Training acc: ", train_acc_list_trfm[np.argmax(val_acc_list_trfm)],
    #       " | Epoch: ", np.argmax(val_acc_list_trfm) + 1)

    print("Pretrain_bert: The Best Training loss: ", np.min(training_loss_list), " | Epoch: ",
          np.argmin(training_loss_list) + 1)
    print("Pretrain_bert: The Best Testing loss: ", best_loss, " | Epoch: ", best_epoch)

    # print("The Best Epoch is: ", best_epoch)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)




