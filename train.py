import sys
import argparse
import random
import time
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from modules import VAE, LSTMEncoder, LSTMDecoder
from data import MonoTextData, readdata, get_batches
from logger import Logger
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def log_data_info(data, dset):
    f = open("trmor_data/"+dset+".txt", "w")
    total_inp_len = 0
    total_out_len = 0
    for d in data:
        inpdata = d[1]
        outdata = d[0]
        total_inp_len += len(inpdata)
        total_out_len += len(outdata)
        f.write(' '.join(inpdata)+'\t'+outdata+'\n')
    f.close()
    avg_inp_len = total_inp_len / len(data)
    avg_out_len = total_out_len / len(data)
    print('%s -- size:%.1d, avg_inp_len: %.2f,  avg_out_len: %.2f' % (dset, len(data), avg_inp_len, avg_out_len))
    
def make_data(batchsize):
    # Read data and get batches...
    data, src_vocab = readdata() # data len:69981
    tgt_data = MonoTextData('trmor_data/trmor.train.txt', label=False)
    tgt_vocab = tgt_data.vocab
    with open('trmor_data/tgt_vocab.json', 'w') as file:
        file.write(json.dumps(tgt_vocab.word2id)) 
    trndata = data[:64000]       # 60000
    vlddata = data[64000:]      # 5981
    tstdata = data[:1]          # 1
    log_data_info(trndata, 'trn')
    log_data_info(vlddata, 'val')
    log_data_info(tstdata, 'tst')
    trn_batches, _ = get_batches(trndata, src_vocab, tgt_vocab, batchsize) 
    vld_batches, _ = get_batches(vlddata, src_vocab, tgt_vocab, batchsize) 
    tst_batches, _ = get_batches(tstdata, src_vocab, tgt_vocab, batchsize) 
    return (trn_batches, vld_batches, tst_batches), src_vocab, tgt_vocab

def test(model, batches, mode):
    report_loss = 0
    report_acc = 0
    indices = list(range(len(batches)))
    random.shuffle(indices)
    report_num_words = report_num_sents = 0
    for i, idx in enumerate(indices):
        #x, y = batches[idx]
        y, x = batches[idx]

        # (batchsize, tx)
        x = x.t().to(device)
        # (batchsize, ty)
        y = y.t().to(device)
        
        # not predict start symbol
        batch_size, sent_len = y.size()
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        
        loss, acc = model.s2s_loss(x, y)
        #loss, acc = model.s2s_loss(y, y)
        
        batch_loss = loss.mean(dim=-1)
        report_loss += loss.sum().item()
        report_acc  += acc


    nll = report_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    acc = report_acc / report_num_words
    print('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f' % (mode, nll,  ppl, acc))

    return nll, ppl, acc

def train(data, model, args):
    trn, val, tst = data
    opt = optim.Adam(model.parameters(), lr=0.001) 
    opt_dict = {"not_improved": 0, "lr": 0.001, "best_loss": 1e4}
    decay_cnt = 0
    best_loss = 1e4
    best_ppl = 0
    decay_epoch = 5
    lr_decay = 0.5
    max_decay = 100
    clip_grad = 5.0
    print('is_decay... %s' % is_decay)

    for epc in range(args.epochs):
        epoch_loss = 0; epoch_align_loss = 0
        epoch_acc = 0
        indices = list(range(len(trn)))
        random.shuffle(indices)
        report_num_words = report_num_sents = 0
        for i, idx in enumerate(indices):
            opt.zero_grad()
            # x: surface form, y; feature form
            y, x = trn[idx] 
            # (batchsize, tx)
            x = x.t().to(device)
            # (batchsize, ty)
            y = y.t().to(device)
            # not predict start symbol
            batch_size, sent_len = y.size()
            report_num_words += (sent_len - 1) * batch_size
            report_num_sents += batch_size
            loss, acc = model.s2s_loss(x, y)
            #loss, acc = model.s2s_loss(y, y)
            
            batch_loss = loss.mean(dim=-1)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            epoch_loss += loss.sum().item()
            epoch_acc += acc
            
        nll = epoch_loss / report_num_sents
        ppl = np.exp(nll * report_num_sents / report_num_words)
        acc = epoch_acc / report_num_words
        print('epoch: %.1d avg_loss: %.4f, ppl: %.4f, acc: %.4f' % (epc, nll,  ppl, acc))

        # VAL
        model.eval()
        with torch.no_grad():
            nll, ppl, acc = test(model, val, "VAL")
            loss = nll
        
        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            best_ppl = ppl
            torch.save(model.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epc >=15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                if is_decay:
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    model.load_state_dict(torch.load(args.save_path))
                    print('new lr: %f' % opt_dict["lr"])
                    decay_cnt += 1
                    opt = optim.Adam(model.parameters(), lr=opt_dict["lr"])
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        model.train()

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.ni = 512
args.enc_nh = 1024
args.dec_nh = 1024
args.nz = 32
args.batchsize = 128
args.epochs = 150
modelname = '64k_trmor_surf2feat_from_ae_frozen_encoder'
is_decay = True
if is_decay: modelname += '_decay'
args.save_path = 'models/'+modelname+'.pt'
args.log_path = 'logs/'+modelname+'.log'
args.device = 'cuda'

args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.5
args.dec_dropout_out = 0.5
sys.stdout = Logger(args.log_path)

# DATA
# src_vocab: feature form -vocabsize 190, tgt_vocab: surface form -vocabsize 76
data, src_vocab, tgt_vocab = make_data(args.batchsize)

# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)

encoder = LSTMEncoder(args, len(tgt_vocab), model_init, emb_init) # len(src_vocab.word2idx)
decoder = LSTMDecoder(args, tgt_vocab, model_init, emb_init) # tgt vocab
model = VAE(encoder, decoder, args)
model.encoder.mode = 's2s'
# print('Model weights from scratch...')
# model.load_state_dict(torch.load('models/4k_trmor_surf2feat_from_ae_v2_frozen_encoder.pt'))

# ae
model.load_state_dict(torch.load('models/trmor_ae_v2.pt'))
print('Model weights loaded from ae...')

# vae
# model.load_state_dict(torch.load('models/trmor_agg1_kls0.10_warm10_0_36.pt'))
# print('Model weights loaded from vae...')

# reset all encoder params
# args.enc_nh = 256
# model.encoder = LSTMEncoder(args, len(src_vocab.word2idx), model_init, emb_init)
# model.encoder.mode = 's2s'
# print('Encoder parameters have been reset...')

# reset all decoder params
model.decoder = LSTMDecoder(args, src_vocab, model_init, emb_init)
print('Decoder parameters have been reset...')

# freeze decoder
# for param in model.decoder.parameters():
#     param.requires_grad = False

# freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False
print('Encoder parameters have been frozen...')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', count_parameters(model))

print(model)
# TRAIN
model.to(args.device)
train(data, model, args)


'''
# PREDICT
_, val, tst = data
model.eval()   
# nll, ppl, acc = test(model, val, "VAL")
with open('pred.txt', "w") as fout:
    for (y, x) in val:
        decoded_batch = model.reconstruct(x.t())
        for sent in decoded_batch:
            fout.write(" ".join(sent) + "\n")
'''