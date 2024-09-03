import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from enumerator import SmilesEnumerator
from utils import split
from torch.nn.functional import one_hot
from random import *
# from sklearn.preprocessing import OneHotEncoder
import copy

PAD = 0
MAX_LEN = 500


class Randomizer(object):

    def __init__(self):
        self.sme = SmilesEnumerator()
        # self.Max_Len = MAX_LEN

    def __call__(self, sm):
        # print(self.Max_Len)
        sm_r = self.sme.randomize_smiles(sm) # Random transform
        if sm_r is None:
            sm_spaced = split(sm) # Spacing
        else:
            sm_spaced = split(sm_r) # Spacing
        sm_split = sm_spaced.split()
        if len(sm_split)<=MAX_LEN - 2:
            return sm_split # List
        else:
            return split(sm).split()

    def random_transform(self, sm):
        '''
        function: Random transformation for SMILES. It may take some time.
        input: A SMILES
        output: A randomized SMILES
        '''
        return self.sme.randomize_smiles(sm)

class Seq2seqDataset(Dataset):

    def __init__(self, train_data, vocab, seq_len, num_classes, max_pred, transform=Randomizer()):
    # def __init__(self, smiles, vocab, seq_len, transform=None):
    #     print(len(data))
        smile1 = []
        smile2 = []
        labels = []
        embed1 = []
        embed2 = []
        for i in range(len(train_data)):
            smile1.append(train_data[i][0])
            smile2.append(train_data[i][2])
            labels.append(train_data[i][1])
            embed1.append(train_data[i][3])
            embed2.append(train_data[i][4])

        # labels_1h = OneHotEncoder(labels)

        self.smile1 = smile1
        self.smile2 = smile2
        self.labels = labels
        self.embed1 = embed1
        self.embed2 = embed2
        self.vocab = vocab
        self.seq_len = seq_len
        self.transform = transform
        self.num_classes = num_classes
        self.max_pred = max_pred

    def __len__(self):
        return len(self.smile1)

    def __getitem__(self, item):
        # print("The value of item: ", item)
        # print("The len of smile1: ", len(self.smile1))
        sm1 = self.smile1[item]
        sm2 = self.smile2[item]
        label = self.labels[item]
        embed1 = self.embed1[item]
        embed2 = self.embed2[item]
        # sm1 = self.transform(sm1) # List
        # sm2 = self.transform(sm2)
        sm1 = split(sm1).split()
        sm2 = split(sm2).split()
        content1_raw = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm1]
        content2_raw = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm2]
        # Truncating bigger sequences:
        if (len(content1_raw) + len(content2_raw)) > 497:
            if len(content1_raw) > 247 and len(content2_raw) > 247:
                content1 = content1_raw[0:247]
                content2 = content2_raw[0:247]
            elif len(content1_raw) > 247 and len(content2_raw) < 247:
                content1 = content1_raw[0:247]
                content2 = content2_raw
            elif len(content1_raw) < 247 and len(content2_raw) > 247:
                content2 = content2_raw[0:247]
                content1 = content1_raw
            else:
                print("Problem in Dataset")
        else:
            content1 = content1_raw
            content2 = content2_raw

        input_ids = [self.vocab.sos_index] + content1 + [self.vocab.sep_index] + content2 +[self.vocab.eos_index]
        input_ids_unmasked = copy.deepcopy(input_ids)
        # padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        segment_ids = [0] * (1 + len(content1) + 1) + [1] * (len(content2) + 1)
        # X.extend(padding)
        # segment_ids.extend(padding)

        # return torch.tensor(X), torch.tensor(segment_ids) ,torch.tensor(label, dtype=torch.long)

        # MASK LM
        n_pred = min(self.max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != self.vocab.sos_index and token != self.vocab.sep_index and token != self.vocab.eos_index]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])

            input_ids[pos] = self.vocab.mask_index  # make mask
            # if random() < 0.8:  # 80%
            #     input_ids[pos] = self.vocab.mask_index  # make mask
            # elif random() < 0.5:  # 10%
            #     index = randint(0, len(self.vocab) - 1)  # random index in vocabulary
            #     input_ids[pos] = self.vocab.stoi[self.vocab.itos[index]]  # replace

        # Zero Paddings
        n_pad = self.seq_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        input_ids_unmasked.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # label_1h = np.zeros((self.num_classes))
        # label_1h[label] += 1

        embed = torch.stack((torch.tensor(embed1),torch.tensor(embed2)), dim=0)

        return torch.tensor(input_ids), torch.tensor(segment_ids), torch.tensor(masked_pos), torch.tensor(
            masked_tokens), torch.tensor(label, dtype=torch.long), torch.tensor(input_ids_unmasked), embed



