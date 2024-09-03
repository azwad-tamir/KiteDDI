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
finetune smiles are uncanonized
Linear header is used to fuse kg_embed and smile_embed
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
# import dataset2
from dataset4 import Seq2seqDataset

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
from rdkit import Chem
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import matthews_corrcoef


seed = 101
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
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
        # self.out3 = nn.Linear(in_features=512, out_features=num_classes)
        # decoder is shared with embedding layer
        embed_weight = self.embed.tok_embed.weight
        n_vocab = vocab_size
        n_dim = d_model
        # n_vocab, n_dim = embed_weight.size()

        self.layer2 = nn.Sequential(nn.Linear(800,512),
                                    nn.LayerNorm(512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.0))

        self.layer3 = nn.Sequential(nn.Linear(1024,512),
                                    nn.LayerNorm(512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.0),
                                    nn.Linear(512,num_classes))

        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos, kg_embed):
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
        kg_embed = kg_embed.flatten(1)
        kg_embed_vec = self.layer2(kg_embed)
        stack_vec = torch.concat((kg_embed_vec,out), dim=1)
        logits_clsf = self.layer3(stack_vec)

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
    parser.add_argument('--n_epoch', '-e', type=int, default=150, help='number of epochs')
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
    parser.add_argument('--lr', type=float, default=5e-5, help='Adam learning rate')
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



def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    acc = 0
    # targets_list = []
    # outputs_list = []
    pred_list = []
    target_list = []
    criterion = nn.CrossEntropyLoss()
    for b, d in enumerate(test_loader):
        input_ids = d[5].cuda()
        segment_ids = d[1].cuda()
        masked_pos = d[2].cuda()
        masked_tokens = d[3].cuda()
        target = d[4].cuda()
        kg_embed = d[6].cuda()

        with torch.no_grad():
            logits_clsf, logits_lm = model(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed)
            # logits_clsf = model(input_ids)

        # loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        # loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, target)  # for sentence classification
        loss = loss_clsf

        total_loss += loss.item()


        pred = torch.max(logits_clsf, axis=1)[1]
        pred_list.extend(pred.detach().cpu().numpy())
        target_list.extend(target.detach().cpu().numpy())
        acc += torch.sum(pred == target).item()

    f1_macro = f1_score(target_list, pred_list, average='macro')
    f1_micro = f1_score(target_list, pred_list, average='micro')
    f1_avg = f1_score(target_list, pred_list, average='weighted')
    f1_bin = matthews_corrcoef(target_list, pred_list)
    auc = 0
    final_loss = total_loss / len(test_loader)
    return final_loss, acc/len(test_loader.dataset), f1_micro, f1_macro, f1_avg, f1_bin, auc, target_list, pred_list
    # return final_loss



train_loss_list = []
eval_loss_list = []
s1_loss_list = []
s2_loss_list = []
train_acc_list = []
val_acc_list = []
s1_acc_list = []
s2_acc_list = []
all_metrices = []
all_results = []

# resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

#     tensor = torch.rand([1, 1, 500, 224])
#     model = ResNet(img_channels=1, num_layers=18, block=BasicBlock, num_classes=1000)
# model_sub = resnet18.ResNet(img_channels=1, num_layers=18, block=resnet18.BasicBlock, num_classes=1000)

def canonize_smiles(smile_data):
    error = 0
    canonized_smiles = []
    for i in range(len(smile_data)):
        if i%1000 == 0:
            print(i)
        first_smile = smile_data[i][0]
        second_smile = smile_data[i][2]
        target = smile_data[i][1]
        try:
            first_new = Chem.CanonSmiles(first_smile)
        except:
            first_new = first_smile
            error+=1

        try:
            second_new = Chem.CanonSmiles(second_smile)
        except:
            second_new = second_smile
            error+=1

        canonized_smiles.append([first_new, target, second_new, smile_data[i][3], smile_data[i][4]])

    return canonized_smiles, error


def add_kg_embed(train_data, kges_dict, db1_drugs):
    train_data_new = []
    db1_names = list(db1_drugs['name'])
    db1_smiles = list(db1_drugs['smiles'])
    for i in range(len(train_data)):
        smile1 = train_data[i][0]
        smile2 = train_data[i][2]
        embed1 = kges_dict[db1_names[db1_smiles.index(smile1)]]
        embed2 = kges_dict[db1_names[db1_smiles.index(smile2)]]
        train_data_new.append([smile1, train_data[i][1], smile2, embed1, embed2])

    return train_data_new


def main():

    # eval_data = []
    args = parse_arguments()
    assert torch.cuda.is_available()
    # args.batch_size = 1

    print('Loading dataset...')
    with open('./data/DB1_data_allFolds', 'rb') as f:
        a = pickle.load(f)

    train_fold, valid_fold, s1_fold, s2_fold = a[0:4]
    train_data_raw = train_fold[0]
    valid_data_raw = valid_fold[0]
    s1_data_raw = s1_fold[0]
    s2_data_raw = s2_fold[0]


    with open('./data/db1_kges_transe.pkl', 'rb') as f:
        kges_dict = pickle.load(f)

    db1_drugs = pd.read_csv("./data/db1_drugs.csv")
    train_data = add_kg_embed(train_data_raw, kges_dict, db1_drugs)
    valid_data = add_kg_embed(valid_data_raw, kges_dict, db1_drugs)
    s1_data = add_kg_embed(s1_data_raw, kges_dict, db1_drugs)
    s2_data = add_kg_embed(s2_data_raw, kges_dict, db1_drugs)


    max_pred = 65
    # Building pretraining Dataset:


    random.seed(101)
    pretrain_df = pd.read_csv('./data/all_smiles1.csv')
    # smiles = list(pretrain_df['canonical_smiles'])
    # pretrain_data = []
    # # ['CN(C)CCOC(=O)C(C1=CC=CC=C1)C1(O)CCCC1', 1, 'CCN(CC)C(C)CN1C2=CC=CC=C2SC2=CC=CC=C12']
    # for i in range(len(smiles)):
    #     pretrain_data.append([smiles[i], 0, smiles[random.randint(0,len(smiles)-1)]])

    ## Canonizing finetune data:
    # train_data_can, error1 = canonize_smiles(train_data)
    # valid_data_can, error2 = canonize_smiles(valid_data)
    # s1_data_can, error3 = canonize_smiles(s1_data)
    # s2_data_can, error4 = canonize_smiles(s2_data)


    vocab = WordVocab.load_vocab(args.vocab)
    dataset_train = Seq2seqDataset(train_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_valid = Seq2seqDataset(valid_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s1 = Seq2seqDataset(s1_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    dataset_s2 = Seq2seqDataset(s2_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    # dataset_test = Seq2seqDataset(train_data_can[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes)
    # dataset_test1 = dataset2.Seq2seqDataset(train_data[0:1], vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)
    # dataset_pretrain = dataset2.Seq2seqDataset(pretrain_data, vocab, seq_len=args.seq_len, num_classes=args.num_classes, max_pred=max_pred)

    # test_size = 10000  # 10000
    # train, test = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    test_loader = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s1_loader = DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    s2_loader = DataLoader(dataset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)
    # pretrain_loader = DataLoader(dataset_pretrain, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)

    # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=args.n_worker)
    # loader_test1 = DataLoader(dataset_test1, batch_size=1, shuffle=True, num_workers=args.n_worker)

    print('Train size:', len(dataset_train))
    print('Test size:', len(dataset_valid))
    print('s1 size:', len(dataset_s1))
    print('s2 size:', len(dataset_s2))
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

    # Loading pretrained model:
    model_bert.load_state_dict(torch.load("./model/bert_pretrain_vocab3_904_2247.pkl"), strict=False)

    optimizer_bert = optim.Adam(model_bert.parameters(), lr=args.lr, weight_decay=1e-5) # Add weight decay weight_decay=1e-5 for L2 regularization
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # Add weight decay weight_decay=1e-5 for L2 regularization
    criterion_bert = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    # print(model)
    # print('Total parameters:', sum(p.numel() for p in model.parameters()))



    best_loss = None
    best_epoch = 0
    best_val_acc = 0
    for e in range(1, args.n_epoch):
        print(">>> Epoch:  ", e)
        for b, d in tqdm(enumerate(train_loader)):
            # break
            input_ids = d[5].cuda()
            segment_ids = d[1].cuda()
            masked_pos = d[2].cuda()
            masked_tokens = d[3].cuda()
            target = d[4].cuda()
            kg_embed = d[6].cuda()
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
            logits_clsf, logits_lm = model_bert(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed)
            # loss_lm = criterion_bert(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
            # loss_lm = (loss_lm.float()).mean()
            loss_clsf = criterion_bert(logits_clsf, target)  # for sentence classification
            # loss = loss_clsf
            # loss = F.multi(output, target)
            loss_clsf.backward()
            optimizer_bert.step()
            if b % 100 == 0:
                print('BERT1: Train {:3d}: iter {:5d} | loss {}'.format(e, b, loss_clsf.item()))

        # Evaluating loss for BERT model:
        loss_train, acc_train, f1_micro, f1_macro, f1_avg, f1_bin, auc, gt_train, pred_train = evaluate(model_bert,
                                                                                                        train_loader,
                                                                                                        )
        train_loss_list.append(loss_train)
        train_acc_list.append(acc_train)
        # eval_data.append(data)

        print('BERT: Train {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_train, acc_train, f1_micro, f1_macro, f1_avg,
                                                        f1_bin, auc))

        loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1, f1_bin1, auc, gt_eval, pred_eval = evaluate(model_bert,
                                                                                                      test_loader,
                                                                                                      )
        eval_loss_list.append(loss_val)
        val_acc_list.append(acc_val)

        print('BERT: Val {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1,
                                                        f1_bin1, auc))

        loss_s1, acc_s1, f1_micro2, f1_macro2, f1_avg2, f1_bin2, auc, gt_s1, pred_s1 = evaluate(model_bert,
                                                                                                s1_loader,
                                                                                                )
        s1_loss_list.append(loss_s1)
        s1_acc_list.append(acc_s1)

        print('BERT: s1 {:3d}: iter {:5d} | loss {} | acc{} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_s1, acc_s1, f1_micro2, f1_macro2, f1_avg2,
                                                        f1_bin2, auc))

        loss_s2, acc_s2, f1_micro3, f1_macro3, f1_avg3, f1_bin3, auc, gt_s2, pred_s2 = evaluate(model_bert,
                                                                                                s2_loader,
                                                                                                )
        s2_loss_list.append(loss_s2)
        s2_acc_list.append(acc_s2)

        print('BERT: s2 {:3d}: iter {:5d} | loss {} | acc{} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_s2, acc_s2, f1_micro3, f1_macro3, f1_avg3,
                                                        f1_bin3, auc))

        all_results.append([gt_train, pred_train, gt_eval, pred_eval, gt_s1, pred_s1, gt_s2, pred_s2])

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_epoch = e
            # torch.save(model_bert.state_dict(), './model_complete/bert_complete0_%d_%d.pkl' % (e, b))

    print("The Best Val accuracy: ", max(val_acc_list), " | Training acc: ",
          train_acc_list[np.argmax(val_acc_list)], " | Epoch: ", np.argmax(val_acc_list) + 1)

    print("Best s1 acc: ", max(s1_acc_list), " || Best s2 acc: ", max(s2_acc_list))

    with open(r"./result/r_ablation_bert_pretrain_vocab3_complete0.pkl", "wb") as output_file:
        pickle.dump(all_results, output_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

# prev_state = torch.load("./model/bert_pretrain_vocab3_904_2247.pkl")
#
# model_bert.load_state_dict(torch.load("./model/bert_pretrain_vocab3_904_2247.pkl"), strict=False)
# new_states = model_bert.state_dict()
# all_keys = list(set(list(prev_state.keys()) + list(new_states.keys())))
# common_keys = []
# for i in range(len(all_keys)):
#     if all_keys[i] in list(prev_state.keys()) and all_keys[i] in list(new_states.keys()):
#         common_keys.append(all_keys[i])
#
# for i in range(len(common_keys)):
#     a = new_states[common_keys[i]].detach().cpu().numpy()
#     b = prev_state[common_keys[i]].detach().cpu().numpy()
#     if not np.array_equal(a,b):
#         print(common_keys[i])



