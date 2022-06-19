# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
from unicodedata import bidirectional
import matplotlib.pyplot as plt
from vqvae_kl_bi import VQVAE
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


def test(batches, mode, args, epc, suffix_codes_trn):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0; epoch_kl_loss = 0; 
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
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_codes_list, recon_preds, kl_loss = args.model.loss(surf, epc)
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
        epoch_kl_loss    += kl_loss.sum().item()
        epoch_acc        += acc
    

    for i in range(args.num_dicts):
        vq_inds[i] = len(epoch_quantized_inds[i])
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    vq = epoch_vq_loss / numwords 
    acc = epoch_acc / epoch_num_tokens
    ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)
    kl = epoch_kl_loss / numwords 

    args.logger.write('\n%s ---  avg_loss: %.4f, avg_recon_loss: %.4f, avg_kl_loss: %.4f, avg_vq_loss: %.4f, ppl: %.4f, acc: %.4f, vq_inds: %s, unique_vq_codes: %d, unique_suffix_codes: %d, val_unique_suffix_codes: %d \n' % (mode, loss, recon, kl, vq, ppl, acc, vq_inds, len(vq_codes), len(suffix_codes), val_unique_suffix_code))
    return loss, recon, vq, acc, vq_inds

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
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
        epoch_vq_loss = 0; epoch_recon_loss = 0; epoch_kl_loss = 0; epoch_logdet = 0
        random.shuffle(indices) # this breaks continuity if there is any
        vq_codes = defaultdict(lambda: 0)
        suffix_codes = defaultdict(lambda: 0)
        epoch_wrong_predictions = []; epoch_correct_predictions = []
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx] 
            # (batchsize)
            loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, kl_loss = args.model.loss(surf, epc)
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
            epoch_kl_loss    += kl_loss.sum().item()
            epoch_acc        += acc
        for i in range(args.num_dicts):
            vq_inds[i] = len(epoch_quantized_inds[i])
        
        loss = epoch_loss / numwords 
        recon = epoch_recon_loss / numwords 
        vq = epoch_vq_loss / numwords 
        ppl = np.exp(epoch_recon_loss/ epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        kl = epoch_kl_loss / numwords 
        args.logger.write('\nepoch: %.1d,  avg_loss: %.4f, avg_recon_loss: %.4f, avg_kl_loss: %.4f, avg_vq_loss: %.4f, ppl: %.4f, acc: %.4f, vq_inds: %s, unique_vq_codes: %d, unique_suffix_codes: %d, logdet: %.4f ' % (epc, loss, recon, kl, vq, ppl, acc, vq_inds, len(vq_codes), len(suffix_codes), 0))
        #tensorboard log
        writer.add_scalar('loss/trn', loss, epc)
        writer.add_scalar('loss/recon_loss/trn', recon, epc)
        writer.add_scalar('loss/vq_loss/trn', vq, epc)
        writer.add_scalar('accuracy/trn', acc, epc)
        trn_loss_values.append(loss)
        trn_vq_values.append(vq)
        trn_recon_loss_values.append(recon)
        trn_vq_inds.append(vq_inds)

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

dataset_type = 'V'

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
elif dataset_type == 'IV' or dataset_type == 'V':
    args.trndata  = 'data/sigmorphon2016/zhou_merged'
    args.valdata  = 'data/sigmorphon2016/turkish-task3-dev'
args.tstdata = args.valdata

args.surface_vocab_file = args.trndata
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
args.dec_dropout_in = 0.5; args.dec_dropout_out = 0.5
args.enc_nh = 512;
args.root_linear_h = 128
args.dec_nh = args.root_linear_h #args.enc_nh; 
args.embedding_dim = args.enc_nh
args.beta = 0.5
args.nz = 128; 
args.num_dicts = 8; args.outcat=0; args.incat = args.enc_nh; 
args.num_dicts_tmp = args.num_dicts; args.outcat_tmp=args.outcat; args.incat_tmp = args.incat; args.dec_nh_tmp = args.dec_nh
args.orddict_emb_num =  4
args.model = VQVAE(args, vocab, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)

# tensorboard
# load pretrained ae weights
args.model_prefix = 'bi_kl0.1_'+str(args.num_dicts)+"x"+str(args.orddict_emb_num)+'_dec'+str(args.dec_nh)+'_suffixd'+str(args.incat)+'/'

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
    #ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_datasetIV-train_d512.pt').to('cpu')
    ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_datasetIV_rf-unidict-train_d512.pt').to('cpu')
    #ae_fhs_vectors_bck = torch.load('model/vqvae/results/fhs/fhs_datasetIV-train_bck_d512.pt').to('cpu')
    _model_id  = 'ae_unilstm_zhou'
elif dataset_type == 'V':
    writer = SummaryWriter("runs/training_vqvae/dataset-V/"+ args.model_prefix)
    #ae_fhs_vectors = torch.load('model/vqvae/results/fhs/fhs_datasetV-unidict-train_d512.pt').to('cpu')
    ae_fhs_vectors_fwd = torch.load('model/vqvae/results/fhs/fhs_datasetV-train_fwd_d512.pt').to('cpu')
    ae_fhs_vectors_bck = torch.load('model/vqvae/results/fhs/fhs_datasetV-train_bck_d512.pt').to('cpu')
    _model_id  = 'ae_bilstm_29k_zhou'

_model_path, surf_vocab  = get_model_info(_model_id) 
for i, vq_layer in enumerate(args.model.ord_vq_layers):
    #vq_layer.embedding.weight.data = ae_fhs_vectors[: args.orddict_emb_num, i*args.model.orddict_emb_dim:(i+1)*args.model.orddict_emb_dim]
    vq_layer.embedding.weight.data = ae_fhs_vectors_bck[: args.orddict_emb_num, i*args.model.orddict_emb_dim:(i+1)*args.model.orddict_emb_dim]


# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)


args.num_dicts = 0; args.outcat=0; args.incat = 0; args.dec_nh = args.enc_nh*2; args.bidirectional=True
args.pretrained_model = VQVAE_AE(args, args.surf_vocab, model_init, emb_init, bidirectional=True)
args.pretrained_model.load_state_dict(torch.load(_model_path), strict=False)
args.num_dicts = args.num_dicts_tmp; args.outcat=args.outcat_tmp; args.incat = args.incat_tmp; args.dec_nh = args.dec_nh_tmp

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
