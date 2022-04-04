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
from data.data import build_data
from model.ae.ae import AE
from model.vae.vae import VAE
from model.vqvae.vqvae import VQVAE

from model.charlm.charlm import CharLM
from common.vocab import VocabEntry
from evaluation.probing.ae_probe import AE_Probe
from evaluation.probing.vae_probe import VAE_Probe
from evaluation.probing.charlm_lstm_probe import CharLM_Lstm_Probe
from evaluation.probing.vqvae_probe import VQVAE_Probe

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0;  epoch_num_instances = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    surfs = []; persons = []
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, person = batches[idx] 
        surfs.append(surf)
        persons.append(person)
        loss, (acc,pred_tokens) = args.model.probe_loss(surf, person)
        epoch_num_instances += surf.size(0)
        epoch_loss += loss.item()
        epoch_acc  += acc
    '''with open('_person_val.txt', 'w') as writer:
        for k in range(len(surfs)):
            surf = surfs[k]
            person_info = persons[k]
            for j in range(surf.size(0)):
                writer.write(''.join(surf_vocab.decode_sentence(surf[j]))+ '--->' + person_vocab.id2word(person_info[j].item()) +'\n')'''
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
    best_loss = 1e4
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        rdict= []
        for i in range(512):
            rdict.append(0)
        surfs = []; preds = []; persons = []
        epoch_loss = 0; epoch_acc = 0; epoch_num_instances = 0
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, person = trnbatches[idx]
            if epc==args.epochs -1:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, person, plot= True, ratiodict=rdict)
                if i==len(indices)-1:
                    loss, (acc,pred_tokens) = args.model.probe_loss(surf, person, plot= True, ratiodict=rdict, last_iter=True)
            else:
                loss, (acc,pred_tokens) = args.model.probe_loss(surf, person, plot= False, ratiodict=None)
            surfs.append(surf)
            persons.append(person)
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
        '''with open('_person_trn.txt', 'w') as writer:
            for k in range(len(surfs)):
                surf = surfs[k]
                person_info = persons[k]
                for j in range(surf.size(0)):
                    writer.write(''.join(surf_vocab.decode_sentence(surf[j]))+ '--->' + person_vocab.id2word(person_info[j].item()) +'\n')'''
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
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    plot_curves(args.task, args.mname, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.mname, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    return trn_acc_values, val_acc_values

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_id = 'vae_neu_2'
model_path, model_vocab  = get_model_info(model_id)
args.mname  = model_id +'_probe' 

# training
args.batchsize = 512; args.epochs = 1000
args.opt= 'Adam'; args.lr = 0.01
args.task = 'surf2person'
args.seq_to_no_pad = 'surface'

# data
with open(model_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)

#args.trndata = 'evaluation/probing/person/data/sosimple.new.trn.combined.txt' 
#args.valdata = 'evaluation/probing/person/data/sosimple.new.seenroots.val.txt'
args.trndata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
args.valdata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.seenroots.val.txt'
args.tstdata = args.valdata

args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args, surf_vocab)
_, person_vocab  = vocab
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(tstdata)

# model
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 256; #for ae,vae,charlm
args.enc_nh = 512; args.dec_nh = 512;  #for ae,vae
args.nh = 512 #for ae,vae,charlm
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0 #for ae,vae,charlm
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0 #for ae,vae
args.embedding_dim = args.enc_nh #for vqvae
args.beta = 0.25 #for vqvae
args.num_dicts = 16
args.rootdict_emb_dim = 512  #for vqvae
args.rootdict_emb_num = 50 #for vqvae
args.orddict_emb_num  = 50   #for vqvae

#args.nz = 512   #for ae,vae
#args.pretrained_model = VQVAE(args, surf_vocab, model_init, emb_init, dict_assemble_type='sum')
#args.model = VQVAE_Probe(args, person_vocab, model_init, emb_init)

args.nz = 32   #for ae,vae
args.pretrained_model = VAE(args, surf_vocab, model_init, emb_init)
args.model = VAE_Probe(args, person_vocab, model_init, emb_init)
args.pretrained_model.load_state_dict(torch.load(model_path), strict=False)

for param in args.model.parameters():
    param.requires_grad = False
for param in args.model.linear.parameters():
    param.requires_grad = True

args.model.to(args.device)

# logging
args.modelname = 'evaluation/probing/person/results/training/'+args.mname+'/'+str(len(trndata))+'_instances/'
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
with open(args.modelname+'/person_vocab.json', 'w') as f:
    f.write(json.dumps(person_vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
#args.logger.write(args)
#args.logger.write('\n')

# plotting
args.fig, args.axs = plt.subplots(2, sharex=True)
args.plt_style = pstyle = '-'

# run
trn_acc_values, val_acc_values = train(batches, args)
plt.savefig(args.fig_path)

print('\nBEST PERSON acc value: %.4f' % max(trn_acc_values))
print('\nBEST PERSON val acc: %.4f' % val_acc_values[np.argmax(trn_acc_values)])
print('\nBEST PERSON at epoch acc: %d' % np.argmax(trn_acc_values))