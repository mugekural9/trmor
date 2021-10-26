from modules import VQVAE
from data.vocab import MonoTextData
from torch import optim
from data import build_data
import random
import argparse, matplotlib
from utils import *
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import math

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0;  epoch_num_tokens = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx][0] 
        loss = args.model(surf)
        correct_tokens, num_tokens= loss['Reconstruction_Acc']
        epoch_num_tokens += num_tokens
        epoch_loss       += loss['loss'].sum().item()
        epoch_acc        += correct_tokens
    nll = epoch_loss / args.valsize  
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f  \n' % (mode, nll,  ppl, acc))
    return nll, ppl, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    args.logger.write('\n----- Parameters: -----')
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4; best_ppl = 0
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
   
    for epc in range(args.epochs):
        total_loss = 0; total_tokens = 0
        epoch_loss = 0; epoch_acc = 0; epoch_num_tokens = 0; epoch_recon_loss = 0; epoch_vq_loss = 0
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            #args.model.zero_grad() 
            # (batchsize, t)
            surf = trnbatches[idx][0] 
            loss, ntokens, acc = args.model(surf)
            total_loss += loss
            total_tokens += ntokens
            correct_tokens, _= acc
            epoch_acc        += correct_tokens

            '''
            correct_tokens, num_tokens= loss['Reconstruction_Acc']
            recon_loss = loss['Reconstruction_Loss']
            vq_loss = loss['VQ_Loss']
            batch_loss = loss#['loss']#.sum() / num_tokens # batch update according to what?
            batch_loss.backward()
            opt.step()
            epoch_num_tokens += num_tokens
            epoch_loss       += loss['loss'].item() #.sum().item()
            epoch_recon_loss += recon_loss
            epoch_vq_loss += vq_loss
            '''
        #print("Epoch Step: %d Loss: %f Tokens " % (i, loss / nseqs))
        ppl = math.exp(total_loss / float(total_tokens))
        acc = epoch_acc / total_tokens
        print('Epoch %d, ppl: %.4f, acc: %.4f' % (epc, ppl, acc))
        #nll = epoch_loss / args.trnsize
        #ppl = np.exp(epoch_loss / epoch_num_tokens)
        #trn_loss_values.append(nll)
        #trn_acc_values.append(acc)
        #args.logger.write('\nepoch: %.1d avg_loss: %.4f, recon_loss: %.4f, vq_loss: %.4f,  acc: %.4f \n' % (epc, nll, epoch_recon_loss/args.trnsize, epoch_vq_loss/args.trnsize, acc))
        '''
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, ppl, acc = test(valbatches, "val", args)
            loss = nll
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            best_ppl = ppl
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()'''
    #plot_curves(args.task, args.bmodel, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    #plot_curves(args.task, args.bmodel, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')


# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.batchsize = 128; args.epochs = 1000
args.opt= 'Adam'; args.lr = 0.01
args.task = 'surf2surf'
args.seq_to_no_pad = 'surface'
args.trndata =  'trmor_data/surf/surf.trn.txt'
args.valdata = 'trmor_data/surf/surf.val.txt'
args.tstdata = 'trmor_data/pos/pos.uniqueroots.txt'
args.fig, args.axs = plt.subplots(2, sharex=True)
args.plt_style = ['-']
args.maxtrnsize = 500
args.maxvalsize = 100; args.maxtstsize = 100 
args.bmodel = 'vqvae' 

args.ni = 512
embedding_dim = 512
args.enc_nh = embedding_dim
args.dec_nh = embedding_dim 
args.nz = embedding_dim

args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.0
args.dec_dropout_out = 0.0
args.device = 'cuda'

num_embeddings = 1
beta = 0.25

surface_vocab = MonoTextData('trmor_data/trmor2018.filtered', label=False).vocab
vqvae = VQVAE(args, surface_vocab, embedding_dim, num_embeddings).to(args.device)
args.model = vqvae
rawdata, batches, vocab = build_data(args)
# logging
args.save_path = args.modelname + '/'+ str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname + '/'+ str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname + '/'+ str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
args.logger.write(args.model)
args.logger.write('beta:' + str(beta))

train(batches, args)
plt.savefig(args.fig_path)