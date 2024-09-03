'''
This file trains the KiteDDI model on Dataset2
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
# from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import matthews_corrcoef


seed = 101
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

        self.self_attn2 = nn.MultiheadAttention(400, 1, dropout=0.0)
        self.norm2 = nn.LayerNorm(400)
        self.norm3 = nn.LayerNorm(400)
        self.dropout2 = nn.Dropout(0.3)

        self.layer2 = nn.Sequential(nn.Linear(800,512),
                                    nn.LayerNorm(512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.3))

        self.layer3 = nn.Sequential(nn.Linear(1312,512),
                                    nn.LayerNorm(512),
                                    nn.ReLU(),
                                    # nn.Dropout(0.3),
                                    nn.Linear(512,num_classes))

        # self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        # self.decoder.weight = embed_weight
        # self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos, kg_embed1, kg_embed2):
        # embedded = self.embed(input_ids)
        embedded = self.embed(input_ids, segment_ids)
        # embedded = self.pe(embedded)

        # output = self.embedding(input_ids, segment_ids)
        hidden = self.encoder(embedded)
        hidden = hidden.permute(1,0,2)
        # print(hidden.shape)
        hidden1 = hidden.reshape(hidden.shape[0], 1, hidden.shape[1], hidden.shape[2])


        # output = output.reshape(output.shape[0], 1, output.shape[1], output.shape[2])
        out = self.resnet18(hidden1)

        kg_rep1 = self.norm2(kg_embed1)
        kg_rep2 = self.norm3(kg_embed2)
        kg_rep_total = torch.concat((kg_rep1, kg_rep2), 1)


        total = torch.concat((kg_rep_total, out), dim=1)
        logits_clsf = self.layer3(total)

        return logits_clsf




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
    parser.add_argument('--n_epoch', '-e', type=int, default=18, help='number of epochs')
    parser.add_argument('--num_classes', '-c', type=int, default=100, help='number of classes')
    # parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_db1_drugs_orig.pkl', help='vocabulary (.pkl)')
    parser.add_argument('--vocab', '-v', type=str, default='./data/vocab_all_smiles1.pkl', help='vocabulary (.pkl)')
    # parser.add_argument('--data', '-d', type=str, default='./data/DB1_data_allFolds', help='train corpus (.csv)')
    parser.add_argument('--out-dir', '-o', type=str, default='./result', help='output directory')
    parser.add_argument('--name', '-n', type=str, default='Pretrain_Bert', help='model name')
    parser.add_argument('--seq_len', type=int, default=500, help='maximum length of the paired seqence')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
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

def roc_aupr_score(y_true, y_score, average="macro"):

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average is None:
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
            return binary_metric(y_true, y_score)
        if average == "macro":
            n_classes = y_score.shape[1]
            score = np.zeros(n_classes)
            for c in range(n_classes):
                y_true_c = y_true[c]
                y_score_c = y_score[c]
                score[c] = binary_metric(y_true_c, y_score_c)
            return np.average(score)

    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    acc = 0
    # targets_list = []
    # outputs_list = []
    score_list = []
    pred_list = []
    target_list = []
    criterion = nn.CrossEntropyLoss()
    for b, d in enumerate(test_loader):
        input_ids = d[5].cuda()
        segment_ids = d[1].cuda()
        masked_pos = d[2].cuda()
        masked_tokens = d[3].cuda()
        target = d[4].cuda()
        kg_embed = d[6]
        kg_embed1 = kg_embed[:, 0, :].cuda()
        kg_embed2 = kg_embed[:, 1, :].cuda()

        with torch.no_grad():
            logits_clsf = model(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed1, kg_embed2)


        # loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        # loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, target)  # for sentence classification
        loss = loss_clsf

        total_loss += loss.item()

        score_list.extend(logits_clsf.detach().cpu().numpy())
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
    return final_loss, acc/len(test_loader.dataset), f1_micro, f1_macro, f1_avg, f1_bin, auc, target_list, pred_list, score_list
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


def add_kg_embed(train_drugs_raw, kges_dict, db2_drugs, db2_names_short, kges_dict_old):
    train_data_new = []
    db2_names = list(db2_drugs['name'])
    db2_smiles = list(db2_drugs['smiles'])
    db2_ids = list(db2_drugs['id'])
    ids_removed = []
    train_drugs0 = []
    train_drugs7 = []

    for i in range(len(train_drugs_raw[0])):
        if len(list(set(train_drugs_raw[0][i]))) == 2:
            train_drugs0.append(train_drugs_raw[0][i])
            train_drugs7.append(train_drugs_raw[7][i])

    train_drugs1 = [train_drugs0, train_drugs7]

    for i in range(len(train_drugs1[0])):
        id1 = db2_ids[db2_names.index(train_drugs1[0][i][0])]
        id2 = db2_ids[db2_names.index(train_drugs1[0][i][1])]
        if id1 in db2_names_short and id2 in db2_names_short:
            smile1 = db2_smiles[db2_names.index(train_drugs1[0][i][0])]
            smile2 = db2_smiles[db2_names.index(train_drugs1[0][i][1])]
            embed1 = kges_dict[id1]
            embed2 = kges_dict[id2]
            # embed1 = kges_dict[train_drugs[0][i][0]]
            # embed2 = kges_dict[train_drugs[0][i][1]]
            label = train_drugs1[1][i]
            if int(torch.sum(label)) != 1:
                print("Error in label generation!!!!")
            train_data_new.append([smile1, int(torch.argmax(label)), smile2, embed1, embed2])
        else:
            ids_removed.append([id1, id2])

    return train_data_new

def normalize_2d_l1(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    norm = np.linalg.norm(matrix, 1)
    # normalized matrix
    matrix = matrix/norm
    return matrix

def normalize_2d_l2(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    norm = np.linalg.norm(matrix, 2)
    # normalized matrix
    matrix = matrix/norm
    return matrix

def renorm(x):
    keys = list(x.keys())
    values = list(x.values())

    values_flatten = np.concatenate(values).ravel()
    mean_flatten = np.mean(values_flatten)
    std_flatten = np.std(values_flatten)
    values_norm = []
    values_norm2 = []

    for i in range(len(values)):
        values_norm.append((values[i] - mean_flatten) / std_flatten)

    for i in range(len(values)):
        mean = np.mean(values[i])
        std = np.std(values[i])
        values_norm2.append((values[i] - mean) / std)

    values_norm_dict1 = {}
    values_norm_dict2 = {}
    for i in range(len(values)):
        values_norm_dict1[keys[i]] = values_norm[i]
        values_norm_dict2[keys[i]] = values_norm2[i]

    return [values_norm_dict1, values_norm_dict2]


def main():

    # eval_data = []
    args = parse_arguments()
    assert torch.cuda.is_available()
    # args.batch_size = 1

    print('Loading dataset...')
    # triplets, unique_dict, weaves, mpnns, afps, vecs, kges, labels = data
    with open('./data/tup_list_db2kges_transe.pkl.pkl', 'rb') as f:
        a = pickle.load(f)

    curr_fold_data = a[0]
    train_drugs = curr_fold_data[0]
    valid_drugs = curr_fold_data[1]
    s1_drugs = curr_fold_data[2]
    s2_drugs = curr_fold_data[3]
    train_drugs_set = []

    for i in range(len(train_drugs[0])):
        train_drugs_set.append(set(train_drugs[0][i]))



    # s2_drugs_new = []
    print("reading s2 drugs")
    s2_temp0 = []
    s2_temp7 = []

    for i in range(len(s2_drugs[0])):
        if set(s2_drugs[0][i]) not in train_drugs_set:
            s2_temp0.append(s2_drugs[0][i])
            s2_temp7.append(s2_drugs[7][i])

    s2_drugs_new = [s2_temp0, s2_drugs[1], s2_drugs[2], s2_drugs[3], s2_drugs[4], s2_drugs[5], s2_drugs[6], s2_temp7]



    with open('./data/db2_kges_transe_new_short.pkl', 'rb') as f:
        kges_dict = pickle.load(f)

    with open('./data/db2_kges_transe.pkl', 'rb') as f:
        kges_dict_old = pickle.load(f)

    with open('./data/db2_names_short.pkl', 'rb') as f:
        db2_names_short = pickle.load(f)

    ####################################################################################################################
    ## Normalizing kges_TransE manually:
    ####################################################################################################################

    x = copy.deepcopy(kges_dict)
    # x_avg = (x-np.mean(x))/np.std(x)

    # with open('../../../data/db1_kges_transe_new1.pkl', 'rb') as f:
    #     x = pickle.load(f)

    keys = list(x.keys())
    values = list(x.values())

    values_flatten = np.concatenate(values).ravel()
    mean_flatten = np.mean(values_flatten)
    std_flatten = np.std(values_flatten)
    values_norm = []
    values_norm2 = []

    for i in range(len(values)):
        values_norm.append((values[i] - mean_flatten) / std_flatten)

    for i in range(len(values)):
        mean = np.mean(values[i])
        std = np.std(values[i])
        values_norm2.append((values[i] - mean) / std)

    kges_transe_l1 = {}
    kges_transe_l2 = {}
    for i in range(len(values)):
        kges_transe_l1[keys[i]] = values_norm[i]
        kges_transe_l2[keys[i]] = values_norm2[i]

    ####################################################################################################################
    ## Normalizing kges_TransE using lin alg:
    ####################################################################################################################
    values_array = np.array(values)
    values_array_trans = np.transpose(values_array)
    normalized_matrix_l1 = normalize_2d_l1(values_array)
    normalized_matrix_l2 = normalize_2d_l2(values_array)
    normalized_matrix_trans_l1 = normalize_2d_l1(values_array_trans)
    normalized_matrix_trans_l2 = normalize_2d_l2(values_array_trans)
    normalized_matrix_l1_flip = np.transpose(normalized_matrix_trans_l1)
    normalized_matrix_l2_flip = np.transpose(normalized_matrix_trans_l2)
    # print("\nSimple Matrix \n", normalized_matrix)

    norm_mat_l1_dict = {}
    norm_mat_l2_dict = {}
    norm_mat_l1_dict_flip = {}
    norm_mat_l2_dict_flip = {}

    # norm_mat_l1_dict_normalized = {}
    # norm_mat_l2_dict_normalized = {}
    # norm_mat_l1_dict_flip_normalized = {}
    # norm_mat_l2_dict_flip_normalized = {}

    for i in range(len(values)):
        norm_mat_l1_dict[keys[i]] = normalized_matrix_l1[i]
        norm_mat_l2_dict[keys[i]] = normalized_matrix_l2[i]
        norm_mat_l1_dict_flip[keys[i]] = normalized_matrix_l1_flip[i]
        norm_mat_l2_dict_flip[keys[i]] = normalized_matrix_l2_flip[i]

    # norm_mat_l1_dict_normalized, _ = renorm(norm_mat_l1_dict)
    # norm_mat_l2_dict_normalized, _ = renorm(norm_mat_l2_dict)
    # norm_mat_l1_dict_flip_normalized, _ = renorm(norm_mat_l1_dict_flip)
    # norm_mat_l2_dict_flip_normalized, _ = renorm(norm_mat_l2_dict_flip)
    kges_transe_l1_linalg = copy.deepcopy(norm_mat_l1_dict)
    kges_transe_l2_linalg = copy.deepcopy(norm_mat_l2_dict)
    ####################################################################################################################


    db2_drugs = pd.read_csv("./data/db2_drugs.csv")
    train_data = add_kg_embed(train_drugs, kges_dict, db2_drugs, db2_names_short, kges_dict_old)
    valid_data = add_kg_embed(valid_drugs, kges_dict, db2_drugs, db2_names_short, kges_dict_old)
    s1_data = add_kg_embed(s1_drugs, kges_dict, db2_drugs, db2_names_short, kges_dict_old)
    s2_data = add_kg_embed(s2_drugs_new, kges_dict, db2_drugs, db2_names_short, kges_dict_old)

    label_dist = np.zeros((100))
    for i in range(len(train_data)):
        label_dist[train_data[i][1]] += 1


    max_pred = 100
    # Building pretraining Dataset:


    random.seed(101)


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


    best_acc = 0
    best_epoch = 0
    best_s2_acc = 0
    for e in range(1, args.n_epoch):
        print(">>> Epoch:  ", e)
        for b, d in tqdm(enumerate(train_loader)):
            # break
            input_ids = d[5].cuda()
            segment_ids = d[1].cuda()
            masked_pos = d[2].cuda()
            masked_tokens = d[3].cuda()
            target = d[4].cuda()
            kg_embed = d[6]
            kg_embed1 = kg_embed[:,0,:].cuda()
            kg_embed2 = kg_embed[:,1,:].cuda()


            # Training Bert model:
            optimizer_bert.zero_grad()
            # logits_clsf = model_bert(torch.t(sm), torch.t(segment_ids))
            logits_clsf = model_bert(torch.t(input_ids), torch.t(segment_ids), masked_pos, kg_embed1, kg_embed2)
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
        loss_train, acc_train, f1_micro, f1_macro, f1_avg, f1_bin, auc, gt_train, pred_train, score_train = evaluate(model_bert,
                                                                                                        train_loader,
                                                                                                        )
        train_loss_list.append(loss_train)
        train_acc_list.append(acc_train)
        # eval_data.append(data)

        print('KITE_DDI: Train {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_weighted {} | MCC {}'.format(e, b, loss_train, acc_train, f1_micro, f1_macro, f1_avg,
                                                        f1_bin))

        loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1, f1_bin1, auc, gt_eval, pred_eval, score_eval = evaluate(model_bert,
                                                                                                      test_loader,
                                                                                                      )
        eval_loss_list.append(loss_val)
        val_acc_list.append(acc_val)

        print('KITE_DDI: Val {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_weighted {} | MCC {}'.format(e, b, loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1,
                                                        f1_bin1))

        loss_U2, acc_U2, f1_micro2, f1_macro2, f1_avg2, f1_bin2, auc, gt_U2, pred_U2, score_U2 = evaluate(model_bert,
                                                                                                s1_loader,
                                                                                                )


        print('KITE_DDI: U2 {:3d}: iter {:5d} | loss {} | acc{} | f1_micro {} | f1_macro {} '
              '| f1_weighted {} | MCC {}'.format(e, b, loss_U2, acc_U2, f1_micro2, f1_macro2, f1_avg2,
                                                        f1_bin2))

        loss_U1, acc_U1, f1_micro3, f1_macro3, f1_avg3, f1_bin3, auc, gt_U1, pred_U1, score_U1 = evaluate(model_bert,
                                                                                                s2_loader,
                                                                                                )


        print('KITE_DDI: U1 {:3d}: iter {:5d} | loss {} | acc{} | f1_micro {} | f1_macro {} '
              '| f1_weighted {} | MCC {}'.format(e, b, loss_U1, acc_U1, f1_micro3, f1_macro3, f1_avg3,
                                                        f1_bin3))

        all_results.append([gt_train, pred_train, score_train, gt_eval, pred_eval, score_eval, gt_U2, pred_U2, score_U2, gt_U1, pred_U1, score_U1])



    # with open(r"./result/r_kiteddi_db2.pkl", "wb") as output_file:
    #     pickle.dump(all_results, output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)





