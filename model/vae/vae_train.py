# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Trainer of character-based variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import matplotlib.pyplot as plt
import numpy as np
from vae import VAE
from common.utils import *
from torch import optim
from data.data import build_data, log_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args, kl_weight):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_kl_loss = 0; epoch_recon_loss = 0
    numwords = args.valsize if mode =='val'  else args.tstsize
    indices = list(range( len(batches)))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx] 
        # (batchsize)
        loss, recon_loss, kl_loss, recon_acc, encoder_fhs = args.model.loss(surf, kl_weight)
        epoch_num_tokens += surf.size(0) * (surf.size(1)-1)  # exclude start token prediction
        epoch_loss       += loss.sum().item()
        epoch_recon_loss += recon_loss.sum().item()
        epoch_kl_loss    += kl_loss.sum().item()
        epoch_acc        += recon_acc
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    kl = epoch_kl_loss / numwords 
    ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- kl_weight: %.2f, avg_loss: %.4f, ppl: %.4f, nll: %.4f, avg_kl_loss: %.4f, acc: %.4f\n' % (mode, kl_weight, loss, ppl, recon, kl, acc))

    return loss, recon, kl, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data

    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    #opt = optim.SGD(filter(lambda p: p.requires_grad, args.model.parameters()), lr=1.0, momentum=0)

    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    
    numbatches = len(trnbatches); indices = list(range(numbatches))
    numwords = args.trnsize
    best_loss = 1e4; trn_loss_values = []; val_loss_values = [];
    trn_kl_values = []; val_kl_values = []
    trn_recon_loss_values = []; val_recon_loss_values = []
    #random.seed(0)
    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * numbatches)
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
        epoch_encoder_fhs = []
        epoch_kl_loss = 0; epoch_recon_loss = 0
        random.shuffle(indices) # this breaks continuity if there is any
        for i, idx in enumerate(indices):
            if args.kl_anneal:
                kl_weight = min(args.kl_max, kl_weight + anneal_rate)
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, kl_loss, acc, encoder_fhs = args.model.loss(surf, kl_weight)
            epoch_encoder_fhs.append(encoder_fhs)
            batch_loss = loss.mean()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(args.model.parameters(),  5.0)
            opt.step()
            epoch_num_tokens += surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.sum().item()
            epoch_recon_loss += recon_loss.sum().item()
            epoch_kl_loss    += kl_loss.sum().item()
            epoch_acc        += acc
        loss = epoch_loss / numwords  
        recon = epoch_recon_loss / numwords
        kl = epoch_kl_loss / numwords
        ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        trn_loss_values.append(loss)
        trn_kl_values.append(kl)
        trn_recon_loss_values.append(recon)
        args.logger.write('\nepoch: %.1d, kl_weight: %.2f, avg_loss: %.4f, ppl: %.4f, nll: %.4f, avg_kl_loss: %.4f, acc: %.4f\n' % (epc, kl_weight, loss, ppl, recon, kl, acc))
        if epc == args.epochs -1:
            epoch_encoder_fhs = torch.cat(epoch_encoder_fhs).squeeze(1)
            torch.save(epoch_encoder_fhs, 'vae_fhs_3487_verbs.pt')

        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon, kl, acc = test(valbatches, "val", args, kl_weight)
        val_loss_values.append(loss)
        val_kl_values.append(kl)
        val_recon_loss_values.append(recon)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
        if epc%5 ==0:
            torch.save(args.model.state_dict(), args.save_path+'_'+str(epc)) # do not save best model but last
        args.model.train()
    plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_kl_values, val_kl_values, args.plt_style, 'kl_loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[2], trn_recon_loss_values, val_recon_loss_values, args.plt_style, 'recon_loss')


# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 50
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vae'
args.seq_to_no_pad = 'surface'
args.kl_start = 0.0
args.kl_anneal = True
args.warm_up = 10
args.kl_max = 1.0
# data
args.trndata = 'data/unlabelled/top50k.wordlist.tur'
args.valdata = 'data/unlabelled/theval.tur'
args.tstdata = args.valdata

args.surface_vocab_file = args.trndata
args.maxtrnsize = 7000000; args.maxvalsize = 3000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'vae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 256; args.nz = 32; 
args.enc_nh = 256; args.dec_nh = 256
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.2; args.dec_dropout_out = 0.0
args.model = VAE(args, vocab, model_init, emb_init)
args.model.to(args.device)
# logging
args.modelname = 'model/'+args.mname+'/results/training/'+str(len(trndata))+'_instances/kl_start'+ \
str(args.kl_start)+'_batchsize'+str(args.batchsize)+'_maxkl_'+str(args.kl_max)+'_warmup'+str(args.warm_up) \
+'_enc_nh' + str(args.enc_nh) \
+'_decdout_in' + str(args.dec_dropout_in) \
+'/'
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
    f.write(json.dumps(vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')
# plotting
args.fig, args.axs = plt.subplots(3)
args.plt_style = pstyle = '-'
args.fig.tight_layout() 

# RUN
train(batches, args)
plt.savefig(args.fig_path)