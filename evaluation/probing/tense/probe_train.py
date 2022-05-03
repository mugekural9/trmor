# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Trainer of tense probe, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from common.utils import *
from common.vocab import VocabEntry
from data.data import build_data
from model.charlm.charlm import CharLM
from model.miniGPT.gpt3 import GPT3

from model.vae.vae import VAE
from model.ae.ae import AE
from model.vqvae.vqvae import VQVAE
from evaluation.probing.charlm_lstm_probe import CharLM_Lstm_Probe
from evaluation.probing.miniGPT_probe import MiniGPT_Probe, MiniGPT_Probe2
from evaluation.probing.vae_probe import VAE_Probe
from evaluation.probing.ae_probe import AE_Probe
from evaluation.probing.vqvae_probe import VQVAE_Probe

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0;  epoch_num_instances = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    surfs = []; tenses = []
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, tense = batches[idx] 
        surfs.append(surf)
        tenses.append(tense)
        loss, (acc, pred_tokens) = args.model.probe_loss(surf, tense)
        epoch_num_instances += surf.size(0)
        epoch_loss += loss.item()
        epoch_acc  += acc
    '''with open('_tense_val.txt', 'w') as writer:
        for k in range(len(surfs)):
            surf = surfs[k]
            tense_info = tenses[k]
            for j in range(surf.size(0)):
                writer.write(''.join(surf_vocab.decode_sentence(surf[j]))+ '--->' + tense_vocab.id2word(tense_info[j].item()) +'\n')'''
    nll = epoch_loss / numbatches
    acc = epoch_acc / epoch_num_instances
    args.logger.write('%s --- avg_loss: %.4f, acc: %.4f  \n' % (mode, nll, acc))
    return nll,  acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.linear.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(len(trnbatches)))
    #random.seed(0)
    best_loss = 1e4
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        rdict= []
        #for i in range(512):
        #    rdict.append(0)
        surfs = []; preds = []; tenses = []
        epoch_loss = 0; epoch_acc = 0; epoch_num_instances = 0
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.linear.zero_grad() 
            # (batchsize, t)
            surf, tense = trnbatches[idx]
            if epc==args.epochs -1:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, tense, plot= False, ratiodict=rdict)
                if i==len(indices)-1:
                    loss, (acc,pred_tokens) = args.model.probe_loss(surf, tense, plot= False, ratiodict=rdict, last_iter=True)
            else:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, tense, plot= False, ratiodict=None)
            surfs.append(surf)
            tenses.append(tense)
            preds.append(pred_tokens)
            loss.backward()
            opt.step()
            epoch_num_instances += surf.size(0) 
            epoch_loss       += loss.item()
            epoch_acc        += acc
        nll = epoch_loss / numbatches
        acc = epoch_acc / epoch_num_instances
        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, acc: %.4f \n' % (epc, nll, acc))
        '''with open('_tense_trn.txt', 'w') as writer:
            for k in range(len(surfs)):
                surf = surfs[k]
                tense_info = tenses[k]
                for j in range(surf.size(0)):
                    writer.write(''.join(surf_vocab.decode_sentence(surf[j]))+ '--->' + tense_vocab.id2word(tense_info[j].item()) +'\n')'''
        '''with open(str(epc)+'_tense_preds.txt', 'w') as writer:
            for k in range(len(surfs)):
                surf = surfs[k]
                pred_tokens = preds[k]
                for j in range(surf.size(0)):
                    writer.write(''.join(surf_vocab.decode_sentence(surf[j]))+ '--->' + tense_vocab.id2word(pred_tokens[j].item()) +'\n')'''
        # VAL
        args.model.linear.eval()
        with torch.no_grad():
            nll, acc = test(valbatches, "val", args)
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if nll < best_loss:
            args.logger.write('update best loss \n')
            best_loss = nll
            torch.save(args.model.state_dict(), args.save_path)
        args.model.linear.train()
    #plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    #plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    return trn_acc_values, val_acc_values

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_id = 'vqvae_best'
model_path, model_vocab  = get_model_info(model_id)
args.mname  = model_id +'_probe' 

# training
args.batchsize = 128; args.epochs = 300
args.opt= 'Adam'; args.lr = 0.01
args.task = 'surf2tense'
args.seq_to_no_pad = 'surface'

# data
with open(model_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)

#args.trndata =  'evaluation/probing/tense/data/tense.uniquesurfs.trn.txt' 
#args.valdata =  'evaluation/probing/tense/data/tense.uniquesurfs.val.txt' 
#args.trndata = 'evaluation/probing/tense/data/sosimple.new.trn.combined.txt' 
#args.valdata = 'evaluation/probing/tense/data/sosimple.new.seenroots.val.txt' 
args.trndata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
args.valdata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.seenroots.val.txt'
args.tstdata = args.valdata

args.maxtrnsize = 700000; args.maxvalsize = 100000; args.maxtstsize = 100000
rawdata, batches, vocab = build_data(args, surf_vocab)
_, tense_vocab  = vocab
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(tstdata)

# pretrained-model
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 256; 
args.enc_nh = 512; 
args.dec_nh = 512;  #for ae,vae
args.nh = 512 #for ae,vae,charlm
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0 
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0 #for ae,vae,vqvae

## miniGPT
num_layers=3
embed_dim=128
num_heads=16
block_size=128
embedding_dropout_rate=0.0; attention_dropout_rate=0.0; residual_dropout_rate=0.0
expand_ratio = 4
args.embed = embed_dim
args.pretrained_model = GPT3(vocab=surf_vocab,
                             num_layers=num_layers,
                             embed_dim=embed_dim,
                             num_heads=num_heads,
                             block_size=block_size,
                             embedding_dropout_rate=embedding_dropout_rate,
                             attention_dropout_rate=attention_dropout_rate,
                             residual_dropout_rate=residual_dropout_rate,
                             expand_ratio=expand_ratio)

## VQVAE
args.beta = 0.25
args.embedding_dim = args.enc_nh
args.rootdict_emb_dim = 512; args.num_dicts = 2; args.nz = 256; args.outcat=0; args.incat = 256
args.rootdict_emb_num = 4000; args.orddict_emb_num  = 500; args.orddict_emb_num_2  = 50
args.pretrained_model = VQVAE(args, surf_vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

## CharLM
#args.pretrained_model = CharLM(args, surf_vocab, model_init, emb_init)

#load pretrained model
args.pretrained_model.load_state_dict(torch.load(model_path), strict=False)

# model
#args.model = CharLM_Lstm_Probe(args, polar_vocab, model_init, emb_init)
args.model = VQVAE_Probe(args, tense_vocab, model_init, emb_init)
#args.model  = MiniGPT_Probe(args, tense_vocab)

for param in args.model.parameters():
    param.requires_grad = False
for param in args.model.linear.parameters():
    param.requires_grad = True

# logging
args.modelname = 'evaluation/probing/tense/results/training/'+args.mname+'/'+str(len(trndata))+'_instances/'
try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(surf_vocab.word2id))
with open(args.modelname+'/tense_vocab.json', 'w') as f:
    f.write(json.dumps(tense_vocab.word2id))

args.model.eval()
args.model.to(args.device)
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# plotting
#args.fig, args.axs = plt.subplots(2, sharex=True)
#args.plt_style = pstyle = '-'

# run
best_trn = []; best_val = []; num_replicas = 5
for i in range(num_replicas):
    args.model.linear.reset_parameters()
    trn_acc_values, val_acc_values = train(batches, args)
    #plt.savefig(args.fig_path)
    best_trn.append(max(trn_acc_values))
    best_val.append(val_acc_values[np.argmax(trn_acc_values)])
    args.logger.write('\n-------------------iter %d-------------------------------------\n' % i)
args.logger.write('num_replicas: \n%d' % num_replicas)
args.logger.write('trn:\n')
args.logger.write('mean: %.4f, std: %.4f \n' % (np.mean(best_trn), np.std(best_trn)))
args.logger.write('val:\n')
args.logger.write('mean: %.4f, std: %.4f \n' % (np.mean(best_val), np.std(best_val)))