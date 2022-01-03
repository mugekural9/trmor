# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Trainer of character-based variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import matplotlib.pyplot as plt
from vae import VAE, VAE_Encoder, VAE_Decoder
from common.utils import *
from torch import optim
from data.data import build_data, log_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args, kl_weight):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_kl_loss = 0; epoch_recon_loss = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx] 
        loss, recon_loss, kl_loss, recon_acc = args.model.loss(surf, kl_weight)
        epoch_num_tokens += surf.size(0) * (surf.size(1)-1)  # exclude start token prediction
        epoch_loss       += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss    += kl_loss.item()
        epoch_acc        += recon_acc
    loss = epoch_loss / numbatches 
    recon = epoch_recon_loss / numbatches 
    kl = epoch_kl_loss / numbatches 
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- kl_weight: %.2f, avg_loss: %.4f, avg_recon_loss: %.4f, avg_kl_loss: %.4f, acc: %.4f\n' % (mode, kl_weight, loss, recon, kl, acc))
    return loss, recon, kl, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data

    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    
    numbatches = len(trnbatches); indices = list(range(numbatches))
    best_loss = 1e4; trn_loss_values = []; val_loss_values = [];
    trn_kl_values = []; val_kl_values = []
    trn_recon_loss_values = []; val_recon_loss_values = []
    random.seed(0)
    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * numbatches)
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
        epoch_kl_loss = 0; epoch_recon_loss = 0
        random.shuffle(indices) # this breaks continuity if there is any
        for i, idx in enumerate(indices):
            if args.kl_anneal:
                kl_weight = min(1.0, kl_weight + anneal_rate)
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, kl_loss, acc = args.model.loss(surf, kl_weight)
            loss.backward()
            opt.step()
            epoch_num_tokens += surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss    += kl_loss.item()
            epoch_acc        += acc
        loss = epoch_loss / numbatches 
        recon = epoch_recon_loss / numbatches 
        kl = epoch_kl_loss / numbatches 
        acc = epoch_acc / epoch_num_tokens
        trn_loss_values.append(loss)
        trn_kl_values.append(kl)
        trn_recon_loss_values.append(recon)
        args.logger.write('\nepoch: %.1d, kl_weight: %.2f, avg_loss: %.4f, avg_recon_loss: %.4f, avg_kl_loss: %.4f, acc: %.4f\n' % (epc, kl_weight, loss, recon, kl, acc))
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
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_kl_values, val_kl_values, args.plt_style, 'kl_loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[2], trn_recon_loss_values, val_recon_loss_values, args.plt_style, 'recon_loss')


# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 10
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vae'
args.seq_to_no_pad = 'surface'
args.kl_start = 0.1
args.kl_anneal = True
args.warm_up = 10
# data
args.trndata = 'model/vae/data/wordlist.tur.trn' # 'model/vae/data/surf.uniquesurfs.trn.txt' 
args.valdata = 'model/vae/data/wordlist.tur.val' # 'model/vae/data/surf.uniquesurfs.val.txt'
args.tstdata = args.valdata
args.surface_vocab_file = args.trndata
args.maxtrnsize = 582000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'vae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 512; args.nz = 32; 
args.enc_nh = 1024; args.dec_nh = 1024
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.5; args.dec_dropout_out = 0.5
args.model = VAE(args, vocab, model_init, emb_init)
args.model.to(args.device)
# logging
args.modelname = 'model/'+args.mname+'/results/training/'+str(len(trndata))+'_instances/'
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