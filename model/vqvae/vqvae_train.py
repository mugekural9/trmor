# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import matplotlib.pyplot as plt
from vqvae import VQVAE

from model.ae.ae import AE
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
torch.autograd.set_detect_anomaly(True)
#tensorboard
writer = SummaryWriter()

def test(batches, mode, args, epc):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0
    numbatches = len(batches)
    indices = list(range(numbatches))
    epoch_quantized_inds = []; vq_inds = []
    for i in range(args.num_dicts):
        epoch_quantized_inds.append(dict())
        vq_inds.append(0)
   
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx] 
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs = args.model.loss(surf, epc)

        if epc % 10 ==0:
            for i in range(surf.shape[0]):             
                args.logger.write('\n')
                args.logger.write(''.join(vocab.decode_sentence(surf[i]))+'----->'+''.join(vocab.decode_sentence(pred_tokens[i])))
        
        for i in range(args.num_dicts):
            for ind in quantized_inds[i][0]:
                ind = ind.item()
                if ind not in epoch_quantized_inds[i]:
                    epoch_quantized_inds[i][ind] = 1
                else:
                    epoch_quantized_inds[i][ind] += 1

        epoch_num_tokens += surf.size(0) * (surf.size(1)-1)  # exclude start token prediction
        epoch_loss       += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_vq_loss    += vq_loss.item()
        epoch_acc        += acc
    
    for i in range(args.num_dicts):
        vq_inds[i] = len(epoch_quantized_inds[i])
    loss = epoch_loss / numbatches 
    recon = epoch_recon_loss / numbatches 
    vq = epoch_vq_loss / numbatches 
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('\n%s ---  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, acc: %.4f, vq_inds: %s \n' % (mode, loss, recon, vq, acc, vq_inds))

    return loss, recon, vq, acc, vq_inds

def train(data, args):
    trnbatches, valbatches, tstbatches = data

    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    #opt = optim.SGD(args.model.parameters(), lr=0.01, momentum=0.9)
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    
    numbatches = len(trnbatches); indices = list(range(numbatches))
    best_loss = 1e4; trn_loss_values = []; val_loss_values = [];
    trn_vq_values = []; val_vq_values = []; 
    trn_vq_inds = []; val_vq_inds = []
    trn_recon_loss_values = []; val_recon_loss_values = []

    for epc in range(args.epochs):
        clusters_list = []; epoch_quantized_inds = []; vq_inds = []
        for i in range(args.num_dicts):
            epoch_quantized_inds.append(dict())
            clusters_list.append(dict())
            vq_inds.append(0)

        epoch_encoder_fhs = []
        epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
        epoch_vq_loss = 0; epoch_recon_loss = 0
        random.shuffle(indices) # this breaks continuity if there is any
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds, encoder_fhs = args.model.loss(surf, epc)

            epoch_encoder_fhs.append(encoder_fhs)
            for i in range(args.num_dicts):
                for ind in quantized_inds[i][0]:
                    if ind not in epoch_quantized_inds[i]:
                        ind = ind.item()
                        epoch_quantized_inds[i][ind] = 1
                    else:
                        epoch_quantized_inds[i][ind] += 1
                _quantized_inds = quantized_inds[i]
                for s in range(surf.shape[0]):
                    ind = _quantized_inds.tolist()[0][s]
                    if ind not in clusters_list[i]:                   
                        clusters_list[i][ind] = []
                    if ''.join(vocab.decode_sentence(surf[s])) not in clusters_list[i][ind]:
                        clusters_list[i][ind].append(''.join(vocab.decode_sentence(surf[s])))

            loss.backward()
            opt.step()
            epoch_num_tokens += torch.sum(surf[:,1:]!=0)#surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss    += vq_loss.item()
            epoch_acc        += acc
        
        for i in range(args.num_dicts):
            vq_inds[i] = len(epoch_quantized_inds[i])

        loss = epoch_loss / numbatches 
        recon = epoch_recon_loss / numbatches 
        vq = epoch_vq_loss / numbatches 
        acc = epoch_acc / epoch_num_tokens
        args.logger.write('\nepoch: %.1d,  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, acc: %.4f, vq_inds: %s ' % (epc, loss, recon, vq, acc, vq_inds))

        #tensorboard log
        writer.add_scalar('loss/trn', loss, epc)
        #writer.add_scalar('num_used_inds/trn', vq_inds, epc)
        writer.add_scalar('loss/recon_loss/trn', recon, epc)
        writer.add_scalar('loss/vq_loss/trn', vq, epc)
        writer.add_scalar('accuracy/trn', acc, epc)

        if epc % 10 == 0:
            for i in range(args.num_dicts):
                with open(str(i)+'_cluster.json', 'w') as json_file:
                    json_object = json.dumps(clusters_list[i], indent = 4, ensure_ascii=False)
                    json_file.write(json_object)

        # Histograms
        # (numinst, hdim)
        epoch_encoder_fhs = torch.cat(epoch_encoder_fhs).squeeze(1)
        fhs_norms =  torch.norm(epoch_encoder_fhs,dim=1)
        fhs_norms = fhs_norms.detach().cpu()
        writer.add_histogram('fhs_norms', fhs_norms, epc)
        
        dict_norms =  torch.norm(args.model.vq_layer_root.embedding.weight,dim=1)
        dict_norms = dict_norms.detach().cpu()
        writer.add_histogram('dict_norms_root', dict_norms, epc)#, bins='auto')
        for i, vq_layer in enumerate(args.model.ord_vq_layers):
            dict_norms =  torch.norm(vq_layer.embedding.weight,dim=1)
            dict_norms = dict_norms.detach().cpu()
            writer.add_histogram('dict_norms_'+str(i), dict_norms, epc)#, bins='auto')


        # Gradient visualizations
        for name,param in args.model.named_parameters():
            if param.requires_grad:
                grad_norms =  torch.norm(param.grad,dim=-1)
                grad_norms = grad_norms.detach().cpu()
                writer.add_histogram('grad_'+name, grad_norms, epc)
        
        trn_loss_values.append(loss)
        trn_vq_values.append(vq)
        trn_recon_loss_values.append(recon)
        trn_vq_inds.append(vq_inds)

        # no matter what save model      
        torch.save(args.model.state_dict(), args.save_path)
        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon, vq, acc, vq_inds = test(valbatches, "val", args, epc)
        
        #tensorboard log
        writer.add_scalar('loss/val', loss, epc)
        #writer.add_scalar('num_used_inds/val', vq_inds, epc)
        writer.add_scalar('loss/recon_loss/val', recon, epc)
        writer.add_scalar('loss/vq_loss/val', vq, epc)
        writer.add_scalar('accuracy/val', acc, epc)

        val_loss_values.append(loss)
        val_vq_values.append(vq)
        val_recon_loss_values.append(recon)
        val_vq_inds.append(vq_inds)
        
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
args.batchsize = 128; args.epochs = 150
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'

# data
args.trndata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
args.valdata = 'model/vqvae/data/trmor2018.uniquesurfs.verbs.seenroots.val.txt'
#args.trndata = 'model/vqvae/data/sosimple.new.trn.combined.txt'
#args.valdata = 'model/vqvae/data/sosimple.new.seenroots.val.txt'
args.tstdata = args.valdata

args.surface_vocab_file = args.trndata
args.maxtrnsize = 100000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 256; 
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.3; args.dec_dropout_out = 0.2
args.enc_nh = 512;
args.dec_nh = args.enc_nh; args.embedding_dim = args.enc_nh; args.nz = args.enc_nh
args.beta = 0.25
args.num_dicts = 7
args.rootdict_emb_dim = 320
args.rootdict_emb_num = 2500
args.orddict_emb_num  = 100
args.model = VQVAE(args, vocab, model_init, emb_init, dict_assemble_type='concat')

# load pretrained ae weights
ae_fhs_vectors = torch.load('ae_004_fhs10k.pt').to('cpu')
args.model.vq_layer_root.embedding.weight.data = ae_fhs_vectors[:args.rootdict_emb_num, :args.rootdict_emb_dim]
for vq_layer in args.model.ord_vq_layers:
    vq_layer.embedding.weight.data = ae_fhs_vectors[:args.orddict_emb_num, :args.model.orddict_emb_dim]

_model_id = 'ae_for_vqvae_004'
_model_path, surf_vocab  = get_model_info(_model_id) 
# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)
args.pretrained_model = VQVAE(args, args.surf_vocab, model_init, emb_init)
args.pretrained_model.load_state_dict(torch.load(_model_path), strict=False)
# CRITIC
args.model.encoder.embed = args.pretrained_model.encoder.embed
args.model.encoder.lstm  = args.pretrained_model.encoder.lstm
args.model.decoder.embed = args.pretrained_model.decoder.embed
args.model.decoder.lstm  = args.pretrained_model.decoder.lstm

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

# RUN
train(batches, args)
writer.close()
