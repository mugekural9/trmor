# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Trainer of surface form pos tagging probe, saves the results under ./results directory.
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
from model.ae.ae import AE
from model.vae.vae import VAE
from model.vqvae.vqvae import VQVAE
from evaluation.probing.charlm_lstm_probe import CharLM_Lstm_Probe
from evaluation.probing.ae_probe import AE_Probe
from evaluation.probing.vae_probe import VAE_Probe
from evaluation.probing.vqvae_probe import VQVAE_Probe

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0;  epoch_num_instances = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    all_pred_tokens = []
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, root_concept = batches[idx] 
        loss, (acc,pred_tokens) = args.model.probe_loss(surf, root_concept)#, args.model_2.linear.weight)
        all_pred_tokens.extend(pred_tokens.squeeze(1).tolist())
        epoch_num_instances += surf.size(0)
        epoch_loss += loss.item()
        epoch_acc  += acc
    nll = epoch_loss / numbatches
    acc = epoch_acc / epoch_num_instances
    args.logger.write('%s --- avg_loss: %.4f, acc: %.4f  \n' % (mode, nll, acc))
    return nll,  acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    #for name, prm in args.model.named_parameters():
    #    args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    #random.seed(0)
    best_loss = 1e4
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        rdict= []
        for i in range(512):
            rdict.append(0)
        epoch_loss = 0; epoch_acc = 0; epoch_num_instances = 0
        random.shuffle(indices) # this breaks continuity if there is
        all_pred_tokens = []
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, root_concept = trnbatches[idx]
            if epc==args.epochs -1:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, root_concept, plot= True, ratiodict=rdict)
                if i==len(indices)-1:
                    loss, (acc,pred_tokens) = args.model.probe_loss(surf, root_concept, plot= True, ratiodict=rdict, last_iter=True)
            else:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, root_concept, plot= False, ratiodict=None)
    
            #loss, (acc,pred_tokens) = args.model.probe_loss(surf, root_concept)   
            #if  ''.join(vocab[0].decode_sentence(surf.squeeze(0))) == '<s>derdim</s>':
            #    breakpoint()
            all_pred_tokens.extend(pred_tokens.squeeze(1).tolist())
            loss.backward()
            opt.step()
            epoch_num_instances += surf.size(0) 
            epoch_loss       += loss.item()
            epoch_acc        += acc
        '''unique_tokens = dict()
        for i in all_pred_tokens:
            if i not in unique_tokens:
                unique_tokens[i] = 1
            else:
                unique_tokens[i] += 1
        print(len(unique_tokens))'''

        nll = epoch_loss / numbatches
        acc = epoch_acc / epoch_num_instances
        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        if acc == 1.0:
            torch.save(args.model.state_dict(), args.save_path)
        #    print('saved model')
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, acc: %.4f \n' % (epc, nll, acc))
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, acc = test(valbatches, "val", args)
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if nll < best_loss:
            args.logger.write('update best loss \n')
            best_loss = nll
        args.model.train()
    plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    return trn_acc_values, val_acc_values

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_id = 'vae_neu_4'
#model_id = 'vqvae_7d_2'

model_path, model_vocab  = get_model_info(model_id)
args.mname  = model_id +'_probe' 

# training
args.batchsize = 512; args.epochs = 1000
args.opt= 'Adam'; args.lr = 0.01
args.task = 'surf2root_concept'
args.seq_to_no_pad = 'surface'

# data
with open(model_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)

args.trndata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
args.valdata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.seenroots.val.txt'
args.tstdata = args.valdata

args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args, surf_vocab)
_, root_concept_vocab  = vocab
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(tstdata)

# model
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 256; #for ae,vae,charlm

args.enc_nh = 512; args.dec_nh = 512;  #for ae,vae
args.nh = 512 #for ae,vae,charlm
args.embedding_dim = args.enc_nh #for vqvae
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0 #for ae,vae,charlm
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0 #for ae,vae
args.beta = 0.25 #for vqvae
args.num_dicts = 7
args.rootdict_emb_dim = 320  #for vqvae
args.rootdict_emb_num = 5000 #for vqvae
args.orddict_emb_num  = 100   #for vqvae

#args.nz = 512   #for ae,vae
#args.pretrained_model = VQVAE(args, surf_vocab, model_init, emb_init, dict_assemble_type='concat')
#args.model = VQVAE_Probe(args, root_concept_vocab, model_init, emb_init)

args.nz = 32
args.pretrained_model = VAE(args, surf_vocab, model_init, emb_init)
args.model = VAE_Probe(args, root_concept_vocab, model_init, emb_init)

#args.pretrained_model.load_state_dict(torch.load(model_path), strict=False)

for param in args.model.parameters():
    param.requires_grad = False
for param in args.model.linear.parameters():
    param.requires_grad = True


args.model.to(args.device)


# logging
args.modelname = 'evaluation/probing/root_concept/results/training/'+args.mname+'/'+str(len(trndata))+'_instances/RANDOM/'
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
with open(args.modelname+'/root_concept_vocab.json', 'w') as f:
    f.write(json.dumps(root_concept_vocab.word2id, ensure_ascii=False))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# plotting
args.fig, args.axs = plt.subplots(2, sharex=True)
args.plt_style = pstyle = '-'

# run
trn_acc_values, val_acc_values = train(batches, args)
plt.savefig(args.fig_path)

print('\nBEST ROOT CONCEPT acc value: %.4f' % max(trn_acc_values))
print('\nBEST ROOT CONCEPT val acc: %.4f' % val_acc_values[np.argmax(trn_acc_values)])
print('\nBEST ROOT CONCEPT  at epoch acc: %d' % np.argmax(trn_acc_values))