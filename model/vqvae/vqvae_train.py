# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import matplotlib.pyplot as plt
from vqvae import VQVAE
from vqvae_ae import VQVAE_AE
from model.vae.vae import VAE

from model.ae.ae import AE
from common.utils import *
from common.vocab import VocabEntry
from torch import optim
from data.data import build_data, log_data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
#reproduce
#torch.manual_seed(0)
#random.seed(0)
torch.autograd.set_detect_anomaly(True)
#tensorboard
#writer = SummaryWriter("runs/training_vqvae/1x3000_5x10/")

def test(batches, mode, args, epc, suffix_codes_trn):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0
    numwords = args.valsize if mode =='val'  else args.tstsize
    numbatches = len(batches)
    indices = list(range(numbatches))
    epoch_quantized_inds = []; vq_inds = []
    for i in range(args.num_dicts):
        epoch_quantized_inds.append(dict())
        vq_inds.append(0)
    vq_codes = defaultdict(lambda: 0)
    suffix_codes = defaultdict(lambda: 0)
    epoch_wrong_predictions = []; epoch_correct_predictions = []
    val_unique_suffix_code = 0
    val_unique_suffix_codes = []
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx] 
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_codes_list, recon_preds,_ = args.model.loss(surf, epc)
        wrong_predictions, correct_predictions = recon_preds
        epoch_wrong_predictions   += wrong_predictions
        epoch_correct_predictions += correct_predictions
        for code in vq_codes_list:
            vq_codes[code] += 1
        for j, code in enumerate(suffix_codes_list):
            suffix_codes[code] += 1
            if code not in suffix_codes_trn:
                val_unique_suffix_code += 1
                val_unique_suffix_codes.append(''.join(vocab.decode_sentence(surf[j])))
        #if epc % 10 ==0:
        #    for i in range(surf.shape[0]):             
        #        args.logger.write('\n')
        #        args.logger.write(''.join(vocab.decode_sentence(surf[i]))+'----->'+''.join(vocab.decode_sentence(pred_tokens[i])))
        for i in range(args.num_dicts):
            for ind in quantized_inds[i][0]:
                ind = ind.item()
                if ind not in epoch_quantized_inds[i]:
                    epoch_quantized_inds[i][ind] = 1
                else:
                    epoch_quantized_inds[i][ind] += 1
        epoch_num_tokens += surf.size(0) * (surf.size(1)-1)  # exclude start token prediction
        epoch_loss       += loss.sum().item()
        epoch_recon_loss += recon_loss.sum().item()
        epoch_vq_loss    += vq_loss.sum().item()
        epoch_acc        += acc
    
    '''# File Operations
    f1 = open("val_wrong_predictions.txt", "w")
    f2 = open("val_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        scode = '-'.join(i.split('dict_code: ')[1].split('-')[1:])
        f1.write(i+'\t'+ str(scode in suffix_codes_trn)+'\n')
    for i in epoch_correct_predictions:
        scode = '-'.join(i.split('dict_code: ')[1].split('-')[1:])
        f2.write(i+'\t'+ str(scode in suffix_codes_trn)+'\n')
    f1.close(); f2.close()'''
    for i in range(args.num_dicts):
        vq_inds[i] = len(epoch_quantized_inds[i])
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    vq = epoch_vq_loss / numwords 
    acc = epoch_acc / epoch_num_tokens
    ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)

    args.logger.write('\n%s ---  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, ppl: %.4f, acc: %.4f, vq_inds: %s, unique_vq_codes: %d, unique_suffix_codes: %d, val_unique_suffix_codes: %d \n' % (mode, loss, recon, vq, ppl, acc, vq_inds, len(vq_codes), len(suffix_codes), val_unique_suffix_code))
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
    numwords = args.trnsize

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
        clusters_list.append(dict())
        epoch_encoder_fhs = []
        epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
        epoch_vq_loss = 0; epoch_recon_loss = 0; epoch_logdet = 0
        random.shuffle(indices) # this breaks continuity if there is any
        vq_codes = defaultdict(lambda: 0)
        suffix_codes = defaultdict(lambda: 0)
        epoch_wrong_predictions = []; epoch_correct_predictions = []
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds, encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, logdet = args.model.loss(surf, epc)
            wrong_predictions, correct_predictions = recon_preds
            epoch_wrong_predictions   += wrong_predictions
            epoch_correct_predictions += correct_predictions
            for code in vq_codes_list:
                vq_codes[code] += 1
            for code in suffix_code_list:
                suffix_codes[code] += 1
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
            
            for s in range(surf.shape[0]):
                ind = suffix_code_list[s]
                if ind not in clusters_list[-1]:                   
                    clusters_list[-1][ind] = []
                if ''.join(vocab.decode_sentence(surf[s])) not in clusters_list[-1][ind]:
                    clusters_list[-1][ind].append(''.join(vocab.decode_sentence(surf[s])))

            batch_loss = loss.mean()
            batch_loss.backward()
            opt.step()
            epoch_num_tokens += surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.sum().item()
            epoch_recon_loss += recon_loss.sum().item()
            epoch_vq_loss    += vq_loss.sum().item()
            epoch_acc        += acc
            epoch_logdet     += logdet
        for i in range(args.num_dicts):
            vq_inds[i] = len(epoch_quantized_inds[i])
        

        '''# File Operations
        f1 = open("trn_wrong_predictions.txt", "w")
        f2 = open("trn_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()'''

        loss = epoch_loss / numwords 
        recon = epoch_recon_loss / numwords 
        vq = epoch_vq_loss / numwords 
        ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        args.logger.write('\nepoch: %.1d,  avg_loss: %.4f, avg_recon_loss: %.4f, avg_vq_loss: %.4f, ppl: %.4f, acc: %.4f, vq_inds: %s, unique_vq_codes: %d, unique_suffix_codes: %d, logdet: %.4f ' % (epc, loss, recon, vq, ppl, acc, vq_inds, len(vq_codes), len(suffix_codes), logdet))
        #tensorboard log
        writer.add_scalar('loss/trn', loss, epc)
        writer.add_scalar('loss/recon_loss/trn', recon, epc)
        writer.add_scalar('loss/vq_loss/trn', vq, epc)
        writer.add_scalar('accuracy/trn', acc, epc)
      
        
        '''# Histograms
        # (numinst, hdim)
        epoch_encoder_fhs = torch.cat(epoch_encoder_fhs).squeeze(1)
        fhs_norms =  torch.norm(epoch_encoder_fhs,dim=1)
        fhs_norms = fhs_norms.detach().cpu()
        writer.add_histogram('fhs_norms', fhs_norms, epc)
        #dict_norms =  torch.norm(args.model.vq_layer_root.embedding.weight,dim=1)
        #dict_norms = dict_norms.detach().cpu()
        writer.add_histogram('dict_norms_root', dict_norms, epc)
        for i, vq_layer in enumerate(args.model.ord_vq_layers):
            dict_norms =  torch.norm(vq_layer.embedding.weight,dim=1)
            dict_norms = dict_norms.detach().cpu()
            writer.add_histogram('dict_norms_'+str(i), dict_norms, epc)'''
        '''# Gradient visualizations
        for name,param in args.model.named_parameters():
            if param.requires_grad:
                grad_norms =  torch.norm(param.grad,dim=-1)
                grad_norms = grad_norms.detach().cpu()
                writer.add_histogram('grad_'+name, grad_norms, epc)'''
        trn_loss_values.append(loss)
        trn_vq_values.append(vq)
        trn_recon_loss_values.append(recon)
        trn_vq_inds.append(vq_inds)
        # no matter what save model      
        torch.save(args.model.state_dict(), args.save_path)
        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon, vq, acc, vq_inds = test(valbatches, "val", args, epc, suffix_codes)
        #tensorboard log
        writer.add_scalar('loss/val', loss, epc)
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
            for i in range(args.num_dicts +1):
                with open(args.modelname+str(i)+'_cluster.json', 'w') as json_file:
                    json_object = json.dumps(clusters_list[i], indent = 4, ensure_ascii=False)
                    json_file.write(json_object)
                with open(args.modelname+str(i)+'_cluster_usage.json', 'w') as wr:
                    for key,val in clusters_list[i].items():
                        usage = {key: len(val)}
                        wr.write(str(usage) + '\n')
        args.model.train()

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 200
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'

dataset_type = 'IV'

# data
if dataset_type == 'I':
    args.trndata = 'data/labelled/verb/trmor2018.uniquesurfs.verb/uniquerooted.trn/trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt'
    args.valdata = 'data/labelled/verb/trmor2018.uniquesurfs.verb/seenroots.val/trmor2018.uniquesurfs.verbs.seenroots.val.txt'
elif dataset_type == 'II':
    args.trndata  = 'data/unlabelled/top50k.wordlist.tur'
    args.valdata  = 'data/unlabelled/theval.tur'
elif dataset_type == 'III':
    args.trndata  = 'data/unlabelled/wordlist.tur'
    args.valdata  = 'data/unlabelled/theval.tur'
elif dataset_type == 'IV':
    args.trndata  = 'data/sigmorphon2016/turkish-task3-train'
    args.valdata  = 'data/sigmorphon2016/turkish-task3-dev'
args.tstdata = args.valdata

args.surface_vocab_file = args.trndata
#args.maxtrnsize = 700000; args.maxvalsize = 5000; args.maxtstsize = 10000
args.maxtrnsize = 10000000; args.maxvalsize = 10000; args.maxtstsize = 10000

rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 256; 
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.2; args.dec_dropout_out = 0.2
args.enc_nh = 512;
args.dec_nh = 128 #args.enc_nh; 
args.embedding_dim = args.enc_nh
args.beta = 0.5
args.rootdict_emb_dim = 512;  args.nz = 128; 
args.num_dicts = 2; args.outcat=0; args.incat = args.enc_nh
args.num_dicts_tmp = args.num_dicts; args.outcat_tmp=args.outcat; args.incat_tmp = args.incat
#args.rootdict_emb_num = 3000
args.orddict_emb_num =  30
args.model = VQVAE(args, vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

# tensorboard
# load pretrained ae weights
args.model_prefix = str(args.num_dicts)+"x"+str(args.orddict_emb_num)+'_suffixd'+str(args.incat)+'/'

if dataset_type == 'I':
    writer = SummaryWriter("runs/training_vqvae/dataset-I/"+ args.model_prefix)
    ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_dataset1-train_fwd_512.pt').to('cpu')
    ae_fhs_vectors_bck = torch.load('model/vqvae/results/fhs/fhs_dataset1-train_bck_512.pt').to('cpu')
    _model_id  = 'ae_011'
elif dataset_type == 'II':
    writer = SummaryWriter("runs/training_vqvae/dataset-II/"+ args.model_prefix)
    ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_top50k_wordlist.tur.pt').to('cpu')
    _model_id  = 'ae_002'
elif dataset_type == 'III':
    writer = SummaryWriter("runs/training_vqvae/dataset-III/"+ args.model_prefix)
    ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_617k_wordlist.tur.pt').to('cpu')
    _model_id  = 'ae_003'
elif dataset_type == 'IV':
    writer = SummaryWriter("runs/training_vqvae/dataset-IV/"+ args.model_prefix)
    ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_datasetIV-train_fwd_d512.pt').to('cpu')
    ae_fhs_vectors_bck = torch.load('model/vqvae/results/fhs/fhs_datasetIV-train_bck_d512.pt').to('cpu')
    _model_id  = 'ae_003'

_model_path, surf_vocab  = get_model_info(_model_id) 
#args.model.vq_layer_root.embedding.weight.data = ae_fhs_vectors[:args.rootdict_emb_num, :args.rootdict_emb_dim]
#args.model.vq_layer_end.embedding.weight.data = ae_fhs_vectors_bck[:args.enddict_emb_num, :args.rootdict_emb_dim]

for i, vq_layer in enumerate(args.model.ord_vq_layers):
    #vq_layer.embedding.weight.data = ae_fhs_vectors_bck[offset_start: offset_end, i*args.model.orddict_emb_dim:(i+1)*args.model.orddict_emb_dim]
    vq_layer.embedding.weight.data = ae_fhs_vectors_bck[:args.orddict_emb_num, i*args.model.orddict_emb_dim:(i+1)*args.model.orddict_emb_dim]


# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)


args.num_dicts = 0; args.outcat=0; args.incat = 0; args.dec_nh = args.enc_nh*2
args.pretrained_model = VQVAE_AE(args, args.surf_vocab, model_init, emb_init)
args.pretrained_model.load_state_dict(torch.load(_model_path), strict=False)
args.num_dicts = args.num_dicts_tmp; args.outcat=args.outcat_tmp; args.incat = args.incat_tmp;  args.dec_nh = 256#args.enc_nh

# CRITIC
args.model.encoder.embed = args.pretrained_model.encoder.embed
args.model.encoder.lstm  = args.pretrained_model.encoder.lstm
#args.model.decoder.embed = args.pretrained_model.decoder.embed
#args.model.decoder.lstm  = args.pretrained_model.decoder.lstm

args.model.to(args.device)
# logging
args.modelname = 'model/'+args.mname+'/results/training/'+str(len(trndata))+'_instances/'+args.model_prefix
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
