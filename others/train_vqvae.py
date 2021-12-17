from modules import VQVAE, VQVAE_w_Attention
from data.vocab import MonoTextData
from torch import optim
from data import build_data, log_data
import random
import argparse, matplotlib
from utils import *
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import math, torch

def test(batches, mode, args):
    total_loss = 0; total_tokens = 0;  
    epoch_acc = 0;  epoch_recon_loss = 0; epoch_vq_loss = 0
    epoch_wrong_predictions = [];
    epoch_correct_predictions = [];
    numbatches = len(batches)
    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx][0] 
        loss = args.model(surf)
        total_loss    += loss['loss']
        epoch_vq_loss += loss['VQ_Loss'].item()
        epoch_recon_loss += loss['Reconstruction_Loss'].item()
        correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = loss['Reconstruction_Acc']
        epoch_acc += correct_tokens
        total_tokens += num_tokens
        epoch_wrong_predictions += wrong_predictions
        epoch_correct_predictions += correct_predictions

    ppl = math.exp(total_loss / float(total_tokens))
    recon_loss = epoch_recon_loss / numbatches
    vq_loss = epoch_vq_loss / numbatches
    acc = epoch_acc / total_tokens
    args.logger.write('\n%s -- ppl: %.4f,  recon_loss: %.4f, vq_loss: %.4f, acc: %.4f ' % (mode, ppl, recon_loss, vq_loss, acc ))
    f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_val_wrong_predictions.txt", "w")
    f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_val_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        f1.write(i+'\n')
    for i in epoch_correct_predictions:
        f2.write(i+'\n')
    f1.close(); f2.close()
    return recon_loss, ppl, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    args.logger.write('\n----- Parameters: -----')
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4;
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []


    for epc in range(args.epochs):
        total_loss = 0; total_tokens = 0
        epoch_acc = 0;  epoch_recon_loss = 0; epoch_vq_loss = 0
        epoch_wrong_predictions = [];
        epoch_correct_predictions = [];
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            # (batchsize, t)
            surf = trnbatches[idx][0] 
            loss = args.model(surf)
            loss['loss'].backward() 
            opt.step()
            opt.zero_grad()
            total_loss    += loss['loss']
            epoch_vq_loss += loss['VQ_Loss'].item()
            epoch_recon_loss += loss['Reconstruction_Loss'].item()

            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = loss['Reconstruction_Acc']
            epoch_acc += correct_tokens
            total_tokens += num_tokens
            epoch_wrong_predictions += wrong_predictions
            epoch_correct_predictions += correct_predictions

        ppl = math.exp(total_loss / float(total_tokens))
        recon_loss = epoch_recon_loss / numbatches
        vq_loss = epoch_vq_loss / numbatches
        acc = epoch_acc / total_tokens
        args.logger.write('\nEpoch %d, ppl: %.4f,  recon_loss: %.4f, vq_loss: %.4f, acc: %.4f' % (epc, ppl, recon_loss, vq_loss, acc))
        trn_loss_values.append(recon_loss)
        trn_acc_values.append(acc)
        f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
        f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()
        '''# VAL
        args.model.eval()
        with torch.no_grad():
            recon_loss, ppl, acc = test(valbatches, "val", args)
        val_loss_values.append(recon_loss)
        val_acc_values.append(acc)
        if recon_loss < best_loss:
            args.logger.write('\nupdate best loss')
            best_loss = recon_loss
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()'''
          
    plot_curves(args.task, args.bmodel, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.bmodel, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')


# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.batchsize = 128; args.epochs = 300
args.opt= 'Adam'; args.lr = 0.001 #0.01
args.task = 'surf2surf'
args.seq_to_no_pad = 'surface'
args.trndata =  'trmor_data/surf/surf.uniquesurfs.trn.txt'
args.valdata =  'trmor_data/surf/surf.uniquesurfs.val.txt'
args.tstdata =  'trmor_data/surf/surf.uniquesurfs.tst.txt'
args.fig, args.axs = plt.subplots(2, sharex=True)
args.plt_style = ['-']
args.maxtrnsize = 57769 
args.maxvalsize = 8329 
args.maxtstsize = 8517
args.bmodel = 'vqvae'
args.surface_vocab_file = args.trndata
embedding_dim = 512
num_embeddings = 10
beta = 0.25
args.device = 'cuda'

surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
vqvae = VQVAE(args, surface_vocab, embedding_dim, num_embeddings, beta).to(args.device)
#vqvae = VQVAE_w_Attention(surface_vocab, embedding_dim, num_embeddings, beta).to(args.device)

args.model = vqvae
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), 0
# logging
args.save_path = args.modelname + '/'+ str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname + '/'+ str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname + '/'+ str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
log_data(trndata, 'trn', vocab, args.logger,args.modelname)
log_data(vlddata, 'val', vocab, args.logger,args.modelname, 'val')
log_data(tstdata, 'tst', vocab, args.logger,args.modelname, 'tst')
args.logger.write(args.model)
args.logger.write('beta:' + str(beta))

train(batches, args)
plt.savefig(args.fig_path)