# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert

'''
Modificaitons:
1 >>> Reducing learning rate to 1e-5
2 >>> Adding dropout of 0.1 in the encoder
3 >>> Increasing d_model to 256 and d_ff to 4*d_model
5 >>> Adding regularization
'''


import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from build_vocab_bert import WordVocab
from dataset1 import Seq2seqDataset
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import matthews_corrcoef

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).to(device)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model).to(device)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model).to(device)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(device)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.lin1 = nn.Linear(self.n_heads * self.d_v, self.d_model).cuda()
        self.lnorm = nn.LayerNorm(self.d_model)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        # print(context)
        # print(context.is_cuda)
        context = context.to(device)

        # output = nn.Linear(self.n_heads * self.d_v, self.d_model)(context)
        output = self.lin1(context)
        # print("TAGLiNE")
        total = output + residual
        return self.lnorm(total), attn
        # return nn.LayerNorm(self.d_model)(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        output = self.fc1(x)
        output = gelu(output)
        # print(output.shape)
        # print(self.fc2)
        output = self.fc2(output)
        return output
        # return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments, n_layers, d_k, d_v, n_heads, d_ff, num_classes):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, maxlen, n_segments)
        # self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, n_heads, d_ff) for _ in range(n_layers)])
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                                        dim_feedforward=d_ff, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        # self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=0) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        output = self.encoder(output)
        # enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # print(enc_self_attn_mask.shape)
        # for layer in self.layers:
        #     output, enc_self_attn = layer(output, enc_self_attn_mask)
            # output = layer(output)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf


def evaluate(model, test_loader, vocab):
    model.eval()
    total_loss = 0
    acc = 0
    # targets_list = []
    # outputs_list = []
    criterion = nn.CrossEntropyLoss()
    pred_list = []
    target_list = []
    for b, d in enumerate(test_loader):
        input_ids = d[0].to(device)
        segment_ids = d[1].to(device)
        masked_pos = d[2].to(device)
        masked_tokens = d[3].to(device)
        target = d[4].to(device)

        with torch.no_grad():
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
        loss_lm = (loss_lm.float()).mean()
        loss_clsf = criterion(logits_clsf, target)  # for sentence classification
        loss = loss_lm + loss_clsf

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



def main():
    # BERT Parameters
    maxlen = 500 # maximum of length
    batch_size = 8
    max_pred = 65  # max tokens of prediction
    n_layers = 6 # number of Encoder of Encoder Layer
    n_heads = 8 # number of heads in Multi-Head Attention
    d_model = 256 # Embedding Size
    d_ff = 4*d_model  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    seq_len = 500
    num_classes = 65
    n_worker = 16
    n_epoch = 150
    learning_rate = 1e-5
    dropout = 0.1


    ####################################################################################################################
    ## Data preparation ################################################################################################
    ####################################################################################################################
    # eval_data = []
    # args = parse_arguments()
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


    vocab = WordVocab.load_vocab('./data/vocab_db1_drugs.pkl')
    vocab_size = len(vocab)
    dataset_train = Seq2seqDataset(train_data, max_pred, vocab, seq_len=seq_len, num_classes=num_classes)
    dataset_valid = Seq2seqDataset(valid_data, max_pred, vocab, seq_len=seq_len, num_classes=num_classes)
    dataset_s1 = Seq2seqDataset(s1_data, max_pred, vocab, seq_len=seq_len, num_classes=num_classes)
    dataset_s2 = Seq2seqDataset(s2_data, max_pred, vocab, seq_len=seq_len, num_classes=num_classes)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    test_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    s1_loader = DataLoader(dataset_s1, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    s2_loader = DataLoader(dataset_s2, batch_size=batch_size, shuffle=True, num_workers=n_worker)
    print('Train size:', len(dataset_train))
    print('Test size:', len(dataset_valid))
    print('s1 size:', len(dataset_s1))
    print('s2 size:', len(dataset_s2))

    ####################################################################################################################

    # a = next(iter(train_loader))
    # masked = []
    # for i in range(len(input_ids[0])):
    #     if input_ids[0][i] == 3:
    #         masked.append(i)

    model = BERT(vocab_size, d_model, maxlen, n_segments, n_layers, d_k, d_v, n_heads, d_ff, num_classes).to(device)
    # print(next(model.parameters()))
    # model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    best_loss = None
    best_epoch = 0
    best_val_acc = 0
    for e in range(1, n_epoch):
        print(">>> Epoch:  ", e)
        for b, d in tqdm(enumerate(train_loader)):
            # break
            input_ids = d[0].to(device)
            segment_ids = d[1].to(device)
            masked_pos = d[2].to(device)
            masked_tokens = d[3].to(device)
            target = d[4].to(device)

            optimizer.zero_grad()
            logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)  # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_clsf = criterion(logits_clsf, target)  # for sentence classification
            loss = loss_lm + loss_clsf

            # loss = F.multi(output, target)
            loss.backward()
            optimizer.step()
            if b % 100 == 0:
                print('Train {:3d}: iter {:5d} | loss {}'.format(e, b, loss.item()))
            # if b % 100 == 0:

        # Evaluating loss for BERT model:
        loss_train, acc_train, f1_micro, f1_macro, f1_avg, f1_bin, auc, gt_train, pred_train = evaluate(model, train_loader, vocab)
        train_loss_list.append(loss_train)
        train_acc_list.append(acc_train)
        # eval_data.append(data)

        print('BERT: Train {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_train, acc_train, f1_micro, f1_macro, f1_avg, f1_bin, auc))

        loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1, f1_bin1, auc, gt_eval, pred_eval = evaluate(model, test_loader, vocab)
        eval_loss_list.append(loss_val)
        val_acc_list.append(acc_val)

        print('BERT: Val {:3d}: iter {:5d} | loss {} | acc {} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_val, acc_val, f1_micro1, f1_macro1, f1_avg1, f1_bin1, auc))

        loss_s1, acc_s1, f1_micro2, f1_macro2, f1_avg2, f1_bin2, auc, gt_s1, pred_s1 = evaluate(model, s1_loader, vocab)
        s1_loss_list.append(loss_s1)
        s1_acc_list.append(acc_s1)

        print('BERT: s1 {:3d}: iter {:5d} | loss {} | acc{} | f1_micro {} | f1_macro {} '
              '| f1_avg {} | f1_bin {} | auc {}'.format(e, b, loss_s1, acc_s1, f1_micro2, f1_macro2, f1_avg2,
                                                        f1_bin2, auc))

        loss_s2, acc_s2, f1_micro3, f1_macro3, f1_avg3, f1_bin3, auc, gt_s2, pred_s2 = evaluate(model, s2_loader, vocab)
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

    with open(r"./result/r_ablation_complete_bert_smile_reg.pkl", "wb") as output_file:
        pickle.dump(all_results, output_file)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)

    # batch = make_batch(sentences, token_list, word_dict, max_pred, vocab_size, maxlen, number_dict)
    # input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    # input_ids = input_ids.to(device)
    # segment_ids = segment_ids.to(device)
    # masked_pos = masked_pos.to(device)
    # isNext = isNext.to(device)
    # masked_tokens = masked_tokens.to(device)
    # for epoch in range(250):
    #     optimizer.zero_grad()
    #     logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    #     loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM
    #     loss_lm = (loss_lm.float()).mean()
    #     loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
    #     loss = loss_lm + loss_clsf
    #     if (epoch + 1) % 10 == 0:
    #         print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    #     loss.backward()
    #     optimizer.step()
    #
    # # Predict mask tokens ans isNext
    # input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    # print(text)
    # print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])
    #
    # logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    # logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    # print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    # print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])
    #
    # logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    # print('isNext : ', True if isNext else False)
    # print('predict isNext : ',True if logits_clsf else False)