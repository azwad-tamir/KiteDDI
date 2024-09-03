'''
Normal Base model with <8 att heads> and <6 Encoder layers>
It runs for 350 Epochs. Training accuracy hits almost 100% while val acc limits at 66%. Bias problem. The model is too big
sq1 and sq2 are stacked to 500 places before inputting them
Use a resnet18 block to process the encoder output
removed the positional encoder layer
Adding regularization in Adam optimizer
'''

'''
Same as bert_pretrain_vocab0 except for smaller lr and larger training epoch
all_smiles1 is used as the pretraining dataset where everything is canonized except for db1 smiles
vocab_sll_smiles1 is used as the vocab file
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
from dataset1_orig import Seq2seqDataset
import dataset2

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
    parser.add_argument('--n_epoch', '-e', type=int, default=1000, help='number of epochs')
    parser.add_argument('--num_classes', '-c', type=int, default=65, help='number of classes')
    # parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_db1_drugs_orig.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_all_smiles1.pkl', help='vocabulary (.pkl)')
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
    pretrain_df = pd.read_csv('./data/all_smiles1.csv')
    smiles = list(pretrain_df['canonical_smiles'])
    pretrain_data = []
    # ['CN(C)CCOC(=O)C(C1=CC=CC=C1)C1(O)CCCC1', 1, 'CCN(CC)C(C)CN1C2=CC=CC=C2SC2=CC=CC=C12']
    for i in range(len(smiles)):
        pretrain_data.append([smiles[i], 0, smiles[random.randint(0,len(smiles)-1)]])


    vocab = WordVocab.load_vocab(args.vocab)
    dataset_pretrain = dataset2.Seq2seqDataset(pretrain_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    test_size = 1000  # 10000
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset_pretrain,[len(dataset_pretrain) - test_size, test_size])
    # dataset_train = Seq2seqDataset(train_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_valid = Seq2seqDataset(valid_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_s1 = Seq2seqDataset(s1_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_s2 = Seq2seqDataset(s2_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_test = Seq2seqDataset(train_data[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_test1 = dataset2.Seq2seqDataset(train_data[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)


    # test_size = 10000  # 10000
    # train, test = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

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


    training_loss_list = []
    best_loss = 1000
    best_epoch = 0
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
            torch.save(model_bert.state_dict(), './model/bert_pretrain_vocab3_%d_%d.pkl' % (e, b))

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




