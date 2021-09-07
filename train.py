import sys
import argparse
import random
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
    # print out data to file
    # f = open("trmor_data/"+dset+".txt", "w")
    total_inp_len = 0
    total_out_len = 0
    for d in data:
        inpdata = d[1]
        outdata = d[0]
        total_inp_len += len(inpdata)
        total_out_len += len(outdata)
        # f.write(' '.join(inpdata)+'\t'+outdata+'\n')
    # f.close()
    avg_inp_len = total_inp_len / len(data)
    avg_out_len = total_out_len / len(data)
    print('%s -- size:%.1d, avg_inp_len: %.2f,  avg_out_len: %.2f' % (dset, len(data), avg_inp_len, avg_out_len))
    
def make_data(trnsize, batchsize):
    # Read data and get batches...
    data, feature_vocab = readdata(trnsize) # data len:69981
    surf_data = MonoTextData('trmor_data/surface.train.txt', label=False) # 60000 surf2surf data from pretraining
    surface_vocab = surf_data.vocab
    with open('trmor_data/surface_vocab.json', 'w') as file:
        file.write(json.dumps(surface_vocab.word2id)) 
    trndata = data[:trnsize]    # trnsize
    vlddata = data[64000:]      # 5981
    tstdata = data[:1]          # 1

    log_data_info(trndata, 'trn')
    trn_batches, _ = get_batches(trndata, surface_vocab, feature_vocab, batchsize) 
   
    log_data_info(vlddata, 'val')
    vld_batches, _ = get_batches(vlddata, surface_vocab, feature_vocab, batchsize) 
   
    log_data_info(tstdata, 'tst')
    tst_batches, _ = get_batches(tstdata, surface_vocab, feature_vocab, batchsize) 

    return (trn_batches, vld_batches, tst_batches), surface_vocab, feature_vocab

def test(model, batches, mode):
    report_loss = 0
    report_acc = 0
    indices = list(range(len(batches)))
    random.shuffle(indices)
    report_num_words = report_num_sents = 0
    for i, idx in enumerate(indices):
        surf, feat= batches[idx]
        # (batchsize, tx)
        surf = surf.t().to(device)
        # (batchsize, ty)
        feat = feat.t().to(device)
        
        # not predict start symbol
        batch_size, sent_len = feat.size()
        report_num_sents += batch_size
        
        #loss, _acc = model.s2s_loss(surf, feat)
        loss, _acc = model.s2s_loss(feat, surf)

        acc, num_words = _acc
        report_num_words += num_words

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
    opt = optim.Adam(model.encoder.parameters(), lr=0.001) 
    #opt = optim.SGD(model.encoder.parameters(), lr=1.0, momentum=0)
    opt_dict = {"not_improved": 0, "lr": 0.001, "best_loss": 1e4}
    decay_cnt = 0
    best_loss = 1e4
    best_ppl = 0
    decay_epoch = 5
    lr_decay = 0.5
    max_decay = 5
    clip_grad = 5.0

    for epc in range(args.epochs):
        epoch_loss = 0; epoch_align_loss = 0
        epoch_acc = 0
        indices = list(range(len(trn)))
        random.shuffle(indices)
        report_num_words = report_num_sents = 0
        for i, idx in enumerate(indices):
            opt.zero_grad()
            # x: surface form, y; feature form
            surf, feat = trn[idx] 
            # (batchsize, tx)
            surf = surf.t().to(device)
            # (batchsize, ty)
            feat = feat.t().to(device)
            # not predict start symbol
            batch_size, sent_len = feat.size()
            report_num_sents += batch_size
           
            #loss, _acc = model.s2s_loss(surf, feat)
            loss, _acc = model.s2s_loss(feat, surf)
            acc, num_words = _acc
            report_num_words += num_words
            
            # loss avg over batch
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
            torch.save(model.state_dict(), save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epc >=15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                if args.is_decay:
                    opt_dict["lr"] = opt_dict["lr"] * lr_decay
                    model.load_state_dict(torch.load(save_path))
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
args.device = 'cuda'
args.ni = 512
args.enc_nh = 1024
args.dec_nh = 1024
args.nz = 32
args.batchsize = 32
args.epochs = 100
args.trnsize = 4000
args.bidirectional = True
args.is_decay = True
args.freeze_encoder = False
args.freeze_decoder = True
args.reset_encoder = True
args.reset_decoder = False
args.task = 'feat2surf'
args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.5
args.dec_dropout_out = 0.5
args.from_pretrained = True
## Switch base model
if args.from_pretrained:
    # ae
    # basemodel = 'models/trmor_agg0_kls0.10_warm10_0_3.pt9'; # model_id = '3_9'
    # vae
    args.basemodel = 'models/trmor_agg1_kls0.10_warm10_0_36.pt'; args.model_id = '0_36'
else:
    args.model_id = 'scratch'
print(args)

modelname = str(int(args.trnsize/1000))+'k_trmor_'+args.task+'_from_'+args.model_id
if args.freeze_encoder: modelname += '_frozen_encoder'
if args.freeze_decoder: modelname += '_frozen_decoder'
if args.bidirectional: modelname += '_bilstm'
if args.is_decay: modelname += '_decay'
save_path = 'models/'+modelname+'.pt'
log_path = 'logs/'+modelname+'.log'
sys.stdout = Logger(log_path)

# DATA
# feature form -vocabsize up to 190, surface form -vocabsize 76
data, surface_vocab, feature_vocab  = make_data(args.trnsize, args.batchsize)

# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init) 
model = VAE(encoder, decoder, args)
model.encoder.mode = 's2s'

if args.from_pretrained:
    model.load_state_dict(torch.load(args.basemodel))
    print('Model weights loaded from ... ', args.basemodel)
else:
    print('Model weights from scratch...')

# reset all encoder params
if args.reset_encoder:
    # args.enc_nh = 256
    model.encoder = LSTMEncoder(args, len(feature_vocab.word2id), model_init, emb_init, bidirectional=args.bidirectional)
    model.encoder.mode = 's2s'
    print('Encoder parameters have been reset...')

# reset all decoder params
if args.reset_decoder:
    model.decoder = LSTMDecoder(args, feature_vocab, model_init, emb_init)
    print('Decoder parameters have been reset...')

# freeze encoder
if args.freeze_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False
    print('Encoder parameters have been frozen...')

# freeze decoder
if args.freeze_decoder:
    for param in model.decoder.parameters():
        param.requires_grad = False
    print('Decoder parameters have been frozen...')

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