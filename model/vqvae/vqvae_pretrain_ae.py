# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import matplotlib.pyplot as plt
from vqvae_ae import VQVAE_AE
from common.utils import *
from common.vocab import VocabEntry
from torch import optim
from data.data import build_data, log_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

#reproduce
#torch.manual_seed(0)
#random.seed(0)

#tensorboard
#writer = SummaryWriter("runs/pretraining_ae/")

def test(batches, mode, args, epc):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0
    numbatches = len(batches)
    numwords = args.valsize if mode =='val'  else args.tstsize

    indices = list(range(numbatches))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx] 
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs = args.model.loss(surf, epc)
        epoch_num_tokens += surf.size(0) * (surf.size(1)-1)  # exclude start token prediction
        epoch_loss       += loss.sum().item()
        epoch_recon_loss += recon_loss.sum().item()
        epoch_vq_loss    += vq_loss.sum().item()
        epoch_acc        += acc
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    vq = epoch_vq_loss / numwords 
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('\n%s ---  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, acc: %.4f\n' % (mode, loss, recon, vq, acc))
    return loss, recon, vq, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data

    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    #opt = optim.SGD(args.model.parameters(), lr=0.01, momentum=0.9)
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    
    numbatches = len(trnbatches); indices = list(range(numbatches))
    numwords = args.trnsize

    best_loss = 1e4; trn_loss_values = []; val_loss_values = [];
    trn_vq_values = []; val_vq_values = []; 
    trn_vq_inds = []; val_vq_inds = []
    trn_recon_loss_values = []; val_recon_loss_values = []
    for epc in range(args.epochs):
        epoch_encoder_fhs = []
        epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
        epoch_vq_loss = 0; epoch_recon_loss = 0
        random.shuffle(indices) # this breaks continuity if there is any
        epoch_rates = []
        epoch_dist_sum = 0
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds, encoder_fhs = args.model.loss(surf, epc)
            batch_loss = loss.mean()
            batch_loss.backward()
            epoch_encoder_fhs.append(encoder_fhs)
            opt.step()
            epoch_num_tokens += torch.sum(surf[:,1:]!=0)#surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.sum().item()
            epoch_recon_loss += recon_loss.sum().item()
            epoch_vq_loss    += vq_loss.sum().item()
            epoch_acc        += acc
        loss = epoch_loss / numwords 
        recon = epoch_recon_loss / numwords 
        vq = epoch_vq_loss / numwords 
        acc = epoch_acc / epoch_num_tokens
        args.logger.write('\nepoch: %.1d,  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, acc: %.4f' % (epc, loss, recon, vq, acc))

        #tensorboard log
        writer.add_scalar('loss/trn', loss, epc)
        writer.add_scalar('loss/recon_loss/trn', recon, epc)
        writer.add_scalar('loss/vq_loss/trn', vq, epc)
        writer.add_scalar('accuracy/trn', acc, epc)
   
        # (numinst, hdim)
        epoch_encoder_fhs = torch.cat(epoch_encoder_fhs).squeeze(1)
        fhs_norms =  torch.norm(epoch_encoder_fhs,dim=1)
        fhs_norms = fhs_norms.detach().cpu()
        writer.add_histogram('fhs_norms', fhs_norms, epc)#, bins='auto')
        if epc == args.epochs -1:
            torch.save(epoch_encoder_fhs, 'fhs_617k_wordlist.tur.pt')

        trn_loss_values.append(loss)
        trn_vq_values.append(vq)
        trn_recon_loss_values.append(recon)

        # no matter what save model      
        torch.save(args.model.state_dict(), args.save_path)

        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon, vq, acc = test(valbatches, "val", args, epc)
        #tensorboard log
        writer.add_scalar('loss/val', loss, epc)
        writer.add_scalar('loss/recon_loss/val', recon, epc)
        writer.add_scalar('loss/vq_loss/val', vq, epc)
        writer.add_scalar('accuracy/val', acc, epc)

        val_loss_values.append(loss)
        val_vq_values.append(vq)
        val_recon_loss_values.append(recon)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 5
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'

# data
#args.trndata = 'data/labelled/verb/trmor2018.uniquesurfs.verb/uniquerooted.trn/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
#args.valdata = 'data/labelled/verb/trmor2018.uniquesurfs.verb/seenroots.val/trmor2018.uniquesurfs.verbs.seenroots.val.txt'

#args.trndata  = 'data/unlabelled/top50k.wordlist.tur'
args.trndata  = 'data/unlabelled/wordlist.tur'
args.valdata  = 'data/unlabelled/theval.tur'
args.tstdata = args.valdata

args.surface_vocab_file = args.trndata
args.maxtrnsize = 1000000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 256; 
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.enc_nh = 512;
args.dec_nh = args.enc_nh; args.embedding_dim = args.enc_nh; args.nz = args.enc_nh
args.beta = 0
args.rootdict_emb_num = 0
args.rootdict_emb_dim = 512; args.num_dicts = 0; args.nz = 512; args.outcat=0; args.incat=0
args.orddict_emb_num  = 0
args.model = VQVAE_AE(args, vocab, model_init, emb_init, dict_assemble_type='concat')
args.model.to(args.device)

#tensorboard
writer = SummaryWriter("runs/pretraining_ae/dataset-III/")


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

# RUN
train(batches, args)
writer.close()
