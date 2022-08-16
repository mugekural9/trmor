# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
from turtle import update
from unicodedata import bidirectional
import matplotlib.pyplot as plt
from model.vqvae.vqvae_kl_bi_early_sup import VQVAE
from vqvae_ae import VQVAE_AE
from model.vae.vae import VAE
import torch.nn.functional as F

from model.ae.ae import AE
from common.utils import *
from common.vocab import VocabEntry
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict, OrderedDict
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
#reproduce
#torch.manual_seed(0)
#random.seed(0)
#torch.autograd.set_detect_anomaly(True)


def test(batches, mode, args, epc, suffix_codes_trn, kl_weight):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0; epoch_kl_loss = 0; 
    numwords = args.valsize if mode =='val'  else args.tstsize
    numbatches = len(batches)
    indices = list(range(numbatches))

    clusters_list = []; epoch_quantized_inds = []; vq_inds = []
    for i in range(args.num_dicts):
        epoch_quantized_inds.append(dict())
        clusters_list.append(dict())
        vq_inds.append(0)
    clusters_list.append(dict())

    new_gen_suffix_codes_val = defaultdict(lambda: 0)
    suffix_codes_val = defaultdict(lambda: 0)
    freq_new_gen_suffix_codes_used = 0
    for i, idx in enumerate(indices):
        # (batchsize, t)
        lxsrc, tags, lxtgt  = batches[idx] 
        #infer mode
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, kl_loss, logdetmaxloss = args.model.loss(lxtgt, tags, kl_weight, epc, mode='val')
       
        ## DICT USAGE TRACKING
        # Keep number of unique codes
        for code in suffix_code_list:
            suffix_codes_val[code] +1
            if code not in suffix_codes_trn:
                new_gen_suffix_codes_val[code] += 1
                freq_new_gen_suffix_codes_used += 1
        # Keep per each dict
        for i in range(args.num_dicts):
            for ind in quantized_inds[i][0]:
                ind = ind.item()
                if ind not in epoch_quantized_inds[i]:
                    epoch_quantized_inds[i][ind] = 1
                else:
                    epoch_quantized_inds[i][ind] += 1
            if epc % 10 ==0:
                # put suffix-code into last cluster
                for s in range(lxtgt.shape[0]):
                    ind = suffix_code_list[s]
                    if ind not in clusters_list[-1]:                   
                        clusters_list[-1][ind] = []
                    if ''.join(args.surf_vocab.decode_sentence(lxtgt[s])) not in clusters_list[-1][ind]:
                        clusters_list[-1][ind].append(''.join(args.surf_vocab.decode_sentence(lxtgt[s])))

        epoch_num_tokens += torch.sum(lxtgt[:,1:] !=0).item()   # exclude start token prediction
        epoch_loss       += loss.sum().item()
        epoch_recon_loss += recon_loss.sum().item()
        epoch_vq_loss    += vq_loss.sum().item()
        epoch_kl_loss    += kl_loss.sum().item()
        epoch_acc        += acc
    
    for i in range(args.num_dicts +1):
        if epc % 10 ==0:
            i = args.num_dicts
            with open(args.modelname+'suffix_codes.json', 'w') as json_file:
                d = OrderedDict(sorted(clusters_list[i].items()))
                json_object = json.dumps(d, indent = 4, ensure_ascii=False)
                json_file.write(json_object)
            with open(args.modelname+'suffix_codes_usage.json', 'w') as wr:
                for key,val in clusters_list[i].items():
                    usage = {key: len(val)}
                    wr.write(str(usage) + '\n')
    for i in range(args.num_dicts):
        vq_inds[i] = len(epoch_quantized_inds[i])
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    vq = epoch_vq_loss / numwords 
    acc = epoch_acc / epoch_num_tokens
    kl = epoch_kl_loss / numwords 
    dict_usage_ratio = len(suffix_codes_val) / (args.orddict_emb_num ** args.num_dicts)
    #tensorboard log
    writer.add_scalar('val/loss', loss, epc)
    writer.add_scalar('val/loss/recon_loss', recon, epc)
    writer.add_scalar('val/loss/kl_loss', kl, epc)
    writer.add_scalar('val/loss/vq_loss', vq, epc)
    writer.add_scalar('val/accuracy', acc, epc)
    writer.add_scalar('val/dict_usage_ratio', dict_usage_ratio, epc)
    writer.add_scalar('val/suffix_codes_val', len(suffix_codes_val), epc)

    args.logger.write('\nVAL')
    args.logger.write('\nloss: %.4f, vq_loss: %.4f, kl_loss: %.4f, recon_loss: %.4f, recon_acc: %.4f' % (loss, vq, kl, recon, acc))
    args.logger.write('\nvq_inds: %s, unique_suffix_codes: %d, dict_usage_ratio: %.4f, new_gen_codes: %d' % (vq_inds, len(suffix_codes_val), dict_usage_ratio, len(new_gen_suffix_codes_val)))
    args.logger.write('\nfreq_new_gen_suffix_codes_used: %d over %d words' % (freq_new_gen_suffix_codes_used, args.valsize))
    return loss, recon, vq, acc, vq_inds

def test_infer(batches, mode, args, epc, suffix_codes_trn, kl_weight):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    epoch_vq_loss = 0; epoch_recon_loss = 0; epoch_kl_loss = 0; 
    numwords = args.valsize if mode =='val'  else args.tstsize
    numbatches = len(batches)
    indices = list(range(numbatches))

    clusters_list = []; epoch_quantized_inds = []; vq_inds = []
    for i in range(args.num_dicts):
        epoch_quantized_inds.append(dict())
        clusters_list.append(dict())
        vq_inds.append(0)
    clusters_list.append(dict())

    new_gen_suffix_codes_val = defaultdict(lambda: 0)
    suffix_codes_val = defaultdict(lambda: 0)
    freq_new_gen_suffix_codes_used = 0
    for i, idx in enumerate(indices):
        # (batchsize, t)
        lxsrc, tags, lxtgt  = batches[idx] 
        #infer mode
        loss, recon_loss, vq_loss, (acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, kl_loss, logdetmaxloss = args.model.loss(lxtgt, None, kl_weight, epc, mode='val')
       
        ## DICT USAGE TRACKING
        # Keep number of unique codes
        for code in suffix_code_list:
            suffix_codes_val[code] +1
            if code not in suffix_codes_trn:
                new_gen_suffix_codes_val[code] += 1
                freq_new_gen_suffix_codes_used += 1
        # Keep per each dict
        for i in range(args.num_dicts):
            for ind in quantized_inds[i][0]:
                ind = ind.item()
                if ind not in epoch_quantized_inds[i]:
                    epoch_quantized_inds[i][ind] = 1
                else:
                    epoch_quantized_inds[i][ind] += 1
            if epc % 10 ==0:
                # put suffix-code into last cluster
                for s in range(lxtgt.shape[0]):
                    ind = suffix_code_list[s]
                    if ind not in clusters_list[-1]:                   
                        clusters_list[-1][ind] = []
                    if ''.join(args.surf_vocab.decode_sentence(lxtgt[s])) not in clusters_list[-1][ind]:
                        clusters_list[-1][ind].append(''.join(args.surf_vocab.decode_sentence(lxtgt[s])))

        epoch_num_tokens += torch.sum(lxtgt[:,1:] !=0).item()   # exclude start token prediction
        epoch_loss       += loss.sum().item()
        epoch_recon_loss += recon_loss.sum().item()
        epoch_vq_loss    += vq_loss.sum().item()
        epoch_kl_loss    += kl_loss.sum().item()
        epoch_acc        += acc
    
    for i in range(args.num_dicts +1):
        if epc % 10 ==0:
            i = args.num_dicts
            with open(args.modelname+'suffix_codes.json', 'w') as json_file:
                d = OrderedDict(sorted(clusters_list[i].items()))
                json_object = json.dumps(d, indent = 4, ensure_ascii=False)
                json_file.write(json_object)
            with open(args.modelname+'suffix_codes_usage.json', 'w') as wr:
                for key,val in clusters_list[i].items():
                    usage = {key: len(val)}
                    wr.write(str(usage) + '\n')
    for i in range(args.num_dicts):
        vq_inds[i] = len(epoch_quantized_inds[i])
    loss = epoch_loss / numwords 
    recon = epoch_recon_loss / numwords 
    vq = epoch_vq_loss / numwords 
    acc = epoch_acc / epoch_num_tokens
    kl = epoch_kl_loss / numwords 
    dict_usage_ratio = len(suffix_codes_val) / (args.orddict_emb_num ** args.num_dicts)
    #tensorboard log
    writer.add_scalar('val_infer/loss', loss, epc)
    writer.add_scalar('val_infer/loss/recon_loss', recon, epc)
    writer.add_scalar('val_infer/loss/kl_loss', kl, epc)
    writer.add_scalar('val_infer/loss/vq_loss', vq, epc)
    writer.add_scalar('val_infer/accuracy', acc, epc)
    writer.add_scalar('val_infer/dict_usage_ratio', dict_usage_ratio, epc)
    writer.add_scalar('val_infer/suffix_codes_val', len(suffix_codes_val), epc)

    args.logger.write('\nVAL INFER')
    args.logger.write('\nloss: %.4f, vq_loss: %.4f, kl_loss: %.4f, recon_loss: %.4f, recon_acc: %.4f' % (loss, vq, kl, recon, acc))
    args.logger.write('\nvq_inds: %s, unique_suffix_codes: %d, dict_usage_ratio: %.4f, new_gen_codes: %d' % (vq_inds, len(suffix_codes_val), dict_usage_ratio, len(new_gen_suffix_codes_val)))
    args.logger.write('\nfreq_new_gen_suffix_codes_used: %d over %d words' % (freq_new_gen_suffix_codes_used, args.valsize))
    return loss, recon, vq, acc, vq_inds

def train(data, args):
    _, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, valbatches ,_, ubatches = data
    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr,  weight_decay=1e-5)
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
   
    numbatches = len(ubatches); indices = list(range(numbatches))
   
    numlxtgtbatches = len(lxtgt_ordered_batches); ltgtindices = list(range(numlxtgtbatches))
    numwords = args.usize
    best_loss = 1e4
    update_ind = 0
    for epc in range(args.epochs):
        args.logger.write('\n-----------------------------------------------------\n')

        clusters_list = []; epoch_quantized_inds = []; vq_inds = []
        for i in range(args.num_dicts):
            epoch_quantized_inds.append(dict())
            clusters_list.append(dict())
            vq_inds.append(0)
        clusters_list.append(dict())
        
        epoch_lxtgt_loss        = 0
        epoch_lxtgt_num_tokens  = 0
        epoch_lxtgt_recon_acc   = 0
        epoch_lxtgt_recon_loss  = 0
        epoch_lxtgt_kl_loss     = 0
        
        epoch_ux_num_tokens = 0   
        epoch_ux_loss       = 0
        epoch_ux_vq_loss    = 0
        epoch_ux_kl_loss    = 0
        epoch_ux_acc        = 0
        epoch_ux_recon_loss = 0
        
        random.shuffle(indices)
        random.shuffle(ltgtindices)

        suffix_codes_trn = defaultdict(lambda: 0)
        for i, idx in enumerate(indices):
            kl_weight = get_kl_weight(update_ind, args.kl_max, 100000.0)
            batch_loss = torch.tensor(0.0).to('cuda')
            update_ind += 1
            args.model.zero_grad()
            # (batchsize, t)
            ux = ubatches[idx]
            ux_loss, ux_recon_loss, ux_vq_loss, (ux_acc,pred_tokens), ux_quantized_inds,  encoder_fhs, vq_codes_list, ux_suffix_code_list, recon_preds, ux_kl_loss, logdetmaxloss = args.model.loss(ux, None, kl_weight, epc)
            if epc>args.ux_start_epc:
                batch_loss += args.ux_weight*(ux_loss.mean())
            if i < len(lxtgt_ordered_batches):
                lidx= ltgtindices[i]
                _, tags, lxtgt  = lxtgt_ordered_batches[lidx] 
                # (batchsize)
                lxtgt_loss, recon_loss, vq_loss, (lxtgt_acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, kl_loss, logdetmaxloss = args.model.loss(lxtgt, tags, kl_weight, epc)
                batch_loss += lxtgt_loss.mean()
            else:
                random.shuffle(ltgtindices) 
                lidx= ltgtindices[0]
                _, tags, lxtgt  = lxtgt_ordered_batches[lidx] 

                # (batchsize)
                lxtgt_loss, recon_loss, vq_loss, (lxtgt_acc,pred_tokens), quantized_inds,  encoder_fhs, vq_codes_list, suffix_code_list, recon_preds, kl_loss, logdetmaxloss = args.model.loss(lxtgt, tags, kl_weight, epc)
                batch_loss += lxtgt_loss.mean()
           
            batch_loss.backward()
            opt.step()
           
            epoch_lxtgt_loss += lxtgt_loss.sum().item()
            epoch_lxtgt_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
            epoch_lxtgt_recon_acc += lxtgt_acc
            epoch_lxtgt_recon_loss += recon_loss.sum().item()
            epoch_lxtgt_kl_loss += kl_loss.sum().item()

            epoch_ux_num_tokens += torch.sum(ux[:,1:] !=0).item() 
            epoch_ux_loss       += ux_loss.sum().item()
            epoch_ux_recon_loss += ux_recon_loss.sum().item()
            epoch_ux_vq_loss    += ux_vq_loss.sum().item()
            epoch_ux_kl_loss    += ux_kl_loss.sum().item()
            epoch_ux_acc        += ux_acc

            ## DICT USAGE TRACKING
            # Keep number of unique codes
            for code in suffix_code_list:
                suffix_codes_trn[code] +=1
            # Keep per each dict
            for i in range(args.num_dicts):
                for ind in quantized_inds[i][0]:
                    ind = ind.item()
                    if ind not in epoch_quantized_inds[i]:
                        epoch_quantized_inds[i][ind] = 1
                    else:
                        epoch_quantized_inds[i][ind] += 1
        for i in range(args.num_dicts):
            vq_inds[i] = len(epoch_quantized_inds[i])
       
        ux_loss  = epoch_ux_loss / numwords 
        ux_recon = epoch_ux_recon_loss / numwords 
        ux_vq    = epoch_ux_vq_loss / numwords 
        ux_acc   = epoch_ux_acc / epoch_ux_num_tokens
        ux_kl    = epoch_ux_kl_loss / numwords 

        dict_usage_ratio = len(suffix_codes_trn) / (args.orddict_emb_num ** args.num_dicts)
        args.logger.write('\nEpoch: %d, kl_weight: %.3f' % (epc, kl_weight))
        args.logger.write('\nTRN')
        args.logger.write('\nux_loss: %.4f, ux_vq_loss: %.4f, ux_kl_loss: %.4f, ux_recon_loss: %.4f, ux_recon_acc: %.4f' % (ux_loss, ux_vq, ux_kl, ux_recon, ux_acc))
        args.logger.write('\nvq_inds: %s, unique_suffix_codes: %d, dict_usage_ratio: %.4f' % ( vq_inds, len(suffix_codes_trn), dict_usage_ratio))

        #tensorboard log
        writer.add_scalar('trn/loss', ux_loss, epc)
        writer.add_scalar('trn/loss/recon_loss', ux_recon, epc)
        writer.add_scalar('trn/loss/kl_loss', ux_kl, epc)
        writer.add_scalar('trn/loss/vq_loss', ux_vq, epc)
        writer.add_scalar('trn/accuracy', ux_acc, epc)
        writer.add_scalar('trn/dict_usage_ratio', dict_usage_ratio, epc)
        writer.add_scalar('trn/suffix_codes_trn', len(suffix_codes_trn), epc)

        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon, vq, acc, vq_inds = test(valbatches, "val", args, epc, suffix_codes_trn, kl_weight)
            _, _, _, _, _ = test_infer(valbatches, "val", args, epc, suffix_codes_trn, kl_weight)
            if loss < best_loss:
                args.logger.write('\nupdate best loss\n')
                best_loss = loss
                torch.save(args.model.state_dict(), args.save_path)
            if epc%3==0 or epc>100:
                shared_acc = shared_task_gen(args,epc) 
                shared_acc_direct = shared_task_gen_direct(args,lxtgt_ordered_batches_TST,epc)
                torch.save(args.model.state_dict(), args.save_path+'_'+str(epc))
                writer.add_scalar('shared-task/accuracy', shared_acc, epc)
                writer.add_scalar('shared-task-direct/accuracy', shared_acc_direct, epc)

        args.model.train()

def shared_task_gen_direct(args, tstbatches,epc):
    i=0
    true=0
    numbatches = len(tstbatches); indices = list(range(numbatches))
    reinflects = []

    with open(args.lang+'_early_sup_DIRECT_'+args.model_prefix[:-1]+'_sharedtask_TRUE.txt', 'w') as writer_true:
        with open(args.lang+'_early_sup_DIRECT_'+args.model_prefix[:-1]+'_sharedtask_FALSE.txt', 'w') as writer_false:
            for _, idx in enumerate(indices):
                if epc<20 and i>1000:
                        break
                i+=1
                # (batchsize, t)
                surf,tags,rsurf = tstbatches[idx] 
                quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss(rsurf, None, 0, 'tst')
                reinflect_tag = [t.item() for t in tags]
                fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(surf)
                root_z = mu.unsqueeze(1)
                root_z = args.model.z_to_dec(root_z)
                bosid = args.surf_vocab.word2id['<s>']
                input = torch.tensor(bosid).to('cuda')
                sft = nn.Softmax(dim=1)
                # Quantized Inputs
                vq_vectors = []
                vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=2:
                    vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=4:
                    vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
                    vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=5:
                    vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=6:
                    vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=8:
                    vq_vectors.append(args.model.ord_vq_layers[6].embedding.weight[reinflect_tag[6]].unsqueeze(0).unsqueeze(0))
                    vq_vectors.append(args.model.ord_vq_layers[7].embedding.weight[reinflect_tag[7]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=10:
                    vq_vectors.append(args.model.ord_vq_layers[8].embedding.weight[reinflect_tag[8]].unsqueeze(0).unsqueeze(0))
                    vq_vectors.append(args.model.ord_vq_layers[9].embedding.weight[reinflect_tag[9]].unsqueeze(0).unsqueeze(0))
                if args.num_dicts >=11:
                    vq_vectors.append(args.model.ord_vq_layers[10].embedding.weight[reinflect_tag[10]].unsqueeze(0).unsqueeze(0))
                
                suffix_z = torch.cat(vq_vectors,dim=2)
                batch_size, seq_len, _ = fhs.size()
                z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)
                c_init = root_z
                h_init = torch.tanh(c_init)
                decoder_hidden = (h_init, c_init)
                MAX_LENGTH = 50
                pred_word = []
                c=0
                pred_form = ''
                while c<MAX_LENGTH:
                    c+=1
                    # (1,1,ni)
                    word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0).to('cuda'))
                    word_embed = torch.cat((word_embed, z_), -1)
                    # output: (1,1,dec_nh)
                    output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
                    # (1, vocab_size)
                    output_logits = args.model.decoder.pred_linear(output).squeeze(1)
                    input = torch.argmax(sft(output_logits))
                    char = args.surf_vocab.id2word(input.item())
                    pred_word.append(char)
                    if char == '</s>':
                        pred_form =''.join(pred_word)
                        break
                gold_form = ''.join(args.surf_vocab.decode_sentence(rsurf.squeeze(0)[1:]))
                if gold_form == pred_form:
                    true +=1
                reinflects.append(gold_form+'\t'+pred_form+'\n')
    acc = (true/i)
    args.logger.write('\nTST reinf acc direct: %.3f' % acc)
    return acc

def shared_task_gen(args,epc):
    i=0
    with open(args.lang+'_early_sup_'+args.model_prefix[:-1]+'_sharedtask_TRUE.txt', 'w') as writer_true:
        with open(args.lang+'_early_sup_'+args.model_prefix[:-1]+'_sharedtask_FALSE.txt', 'w') as writer_false:
            with open(args.tstdata, 'r') as reader:
                true=0; false = 0
                for line in reader:
                    if i>1000:#epc<20 and i>1000:
                        break
                    i+=1
                    split_line = line.strip().split('\t')
                    inflected_word, asked_tag, gold_reinflection  = split_line
                    oracle_keys = oracle(args, asked_tag, inflected_word, gold_reinflection)
                    key = [int(n) for n in oracle_keys.split('-')]
                    reinflected_word, suffix_code =  reinflect(args, inflected_word,key)
                    reinflected_word = reinflected_word[:-4]
                    #writer.write(inflected_word +'\t'+ asked_tag+'\t'+reinflected_word + '\n')
                    if reinflected_word == gold_reinflection:
                        true +=1
                        writer_true.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ '-'.join([str(s) for s in suffix_code])+'\n')
                    else:
                        writer_false.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ '-'.join([str(s) for s in suffix_code])+'\n')
    acc = (true/i)
    args.logger.write('\nShared Task oracle acc: %.3f over %d words' % (acc,i))
    return acc

def oracle(args, asked_tag, inflected_word, reinflected_word, itr=0):
    x = torch.tensor([args.surf_vocab.word2id['<s>']] + args.surf_vocab.encode_sentence(reinflected_word) + [args.surf_vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss(x, None, 0, 'tst')
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])
    return mapped_inds#''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds +'\t---> ' + reinflected_word

def reinflect(args, inflected_word, reinflect_tag):
    x = torch.tensor([args.surf_vocab.word2id['<s>']] + args.surf_vocab.encode_sentence(inflected_word) + [args.surf_vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
    vq_vectors = []
  
    # kl version
    fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(x)
    _root_fhs = mu.unsqueeze(0)
    #_root_fhs = args.model.reparameterize(mu, logvar)
    _root_fhs = args.model.z_to_dec(_root_fhs)

    vq_vectors.append(_root_fhs)

    bosid = args.surf_vocab.word2id['<s>']
    input = torch.tensor(bosid).to('cuda')
    sft = nn.Softmax(dim=1)
    # Quantized Inputs

    vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=2:
        vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=3:
        vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=4:
        vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=5:
        vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=6:
        vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))

    if args.num_dicts >=8:
        vq_vectors.append(args.model.ord_vq_layers[6].embedding.weight[reinflect_tag[6]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[7].embedding.weight[reinflect_tag[7]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=10:
        vq_vectors.append(args.model.ord_vq_layers[8].embedding.weight[reinflect_tag[8]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[9].embedding.weight[reinflect_tag[9]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=11:
        vq_vectors.append(args.model.ord_vq_layers[10].embedding.weight[reinflect_tag[10]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=16:
        vq_vectors.append(args.model.ord_vq_layers[11].embedding.weight[reinflect_tag[11]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[12].embedding.weight[reinflect_tag[12]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[13].embedding.weight[reinflect_tag[13]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[14].embedding.weight[reinflect_tag[14]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[15].embedding.weight[reinflect_tag[15]].unsqueeze(0).unsqueeze(0))

    vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
    root_z, suffix_z = vq_vectors
    batch_size, seq_len, _ = fhs.size()
    z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)

    c_init = root_z
    h_init = torch.tanh(c_init)
    decoder_hidden = (h_init, c_init)
    copied = []; i = 0
    MAX_LENGTH = 50
    c=0
    while c<MAX_LENGTH:
        c+=1
        # (1,1,ni)
        word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0).to('cuda'))
        word_embed = torch.cat((word_embed, z_), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = args.model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits))
        char = args.surf_vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            return(''.join(copied), (reinflect_tag))
    return(''.join(copied), (reinflect_tag))

def get_kl_weight(update_ind, thres, rate):
    upnum = 1500
    if update_ind <= upnum:
        return 0.0
    else:
        w = (1.0/rate)*(update_ind - upnum)
        if w < thres:
            return w
        else:
            return thres

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 301
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'
args.kl_max = 0.2
dataset_type = 'V'
args.lang='spanish'


if dataset_type == 'V':
    _model_id = 'ae_'+args.lang+'_unsup_660'
    #ae_fhs_vectors_fwd = torch.load('model/vqvae/results/fhs/'+args.lang+'_fhs_datasetV-train_fwd_d660.pt').to('cpu')
    ae_fhs_vectors_bck = torch.load('model/vqvae/results/fhs/'+args.lang+'_fhs_datasetV-train_bck_d660.pt').to('cpu')
    #ae_fhs_vectors     = torch.load('model/vqvae/results/fhs/'+args.lang+'_fhs_datasetV-train_all_d1320.pt').to('cpu')

_model_path, surf_vocab  = get_model_info(_model_id) 
# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)

# data
args.trndata  = 'data/sigmorphon2016/'+args.lang+'-task3-train'
args.valdata  = 'data/sigmorphon2016/'+args.lang+'-task3-test'
args.tstdata  = args.valdata
args.unlabeled_data = 'data/sigmorphon2016/'+args.lang+'_ux_ctrl.txt'
args.maxtrnsize = 1000000; args.maxvalsize = 1000; args.maxtstsize = 1000000000000; args.maxusize=1000000
args.ux_weight = 0.7
args.ux_start_epc = 20

if args.lang == 'finnish':
    from data.data_2_finnish import build_data
elif args.lang == 'turkish':
    from data.data_2_turkish import build_data
elif args.lang == 'hungarian':
    from data.data_2_hungarian import build_data
elif args.lang == 'maltese':
    from data.data_2_maltese import build_data
elif args.lang == 'navajo':
    from data.data_2_navajo import build_data
elif args.lang == 'russian':
    from data.data_2_russian import build_data
elif args.lang == 'arabic':
    from data.data_2_arabic import build_data
elif args.lang == 'german':
    from data.data_2_german import build_data
elif args.lang == 'spanish':
    from data.data_2_spanish import build_data
#elif args.lang == 'georgian':
#    from data.data_2_georgian import build_data


rawdata, batches, _, tag_vocabs = build_data(args, args.surf_vocab)
trndata, valdata, tstdata, udata = rawdata
args.trnsize , args.valsize, args.tstsize, args.usize = len(trndata), len(valdata), len(tstdata), len(udata)
# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.3; args.dec_dropout_out = 0.0
args.ni = 256; 
args.enc_nh = 660;
args.dec_nh = 256  
args.embedding_dim = args.enc_nh
args.beta = 0.1
args.nz = 128; 
args.num_dicts = len(tag_vocabs)
args.outcat=0; 
args.orddict_emb_num = 1
args.incat = args.enc_nh; 

args.num_dicts_tmp = args.num_dicts; args.outcat_tmp=args.outcat; args.incat_tmp = args.incat; args.dec_nh_tmp = args.dec_nh
args.model = VQVAE(args, args.surf_vocab,  tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)

# tensorboard
# load pretrained ae weights
args.model_prefix = 'batchsize'+str(args.batchsize)+'_beta'+str(args.beta)+'_bi_kl'+str(args.kl_max)+'_'+str(args.num_dicts)+"x"+str(args.orddict_emb_num)+'_dec'+str(args.dec_nh)+'_suffixd'+str(args.incat)+'/'
writer = SummaryWriter("runs/early-supervision/"+args.lang+'/'+ args.model_prefix)

for i, vq_layer in enumerate(args.model.ord_vq_layers):
    vq_layer.embedding.weight.data = ae_fhs_vectors_bck[:vq_layer.embedding.weight.size(0), i*args.model.orddict_emb_dim:(i+1)*args.model.orddict_emb_dim]


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
args.modelname = 'model/'+args.mname+'/results/training/'+args.lang+'/early-supervision/'+str(args.lxtgtsize)+'_instances/'+args.model_prefix

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
    f.write(json.dumps(args.surf_vocab.word2id))

j=0
for key,val in tag_vocabs.items():
    with open(args.modelname+'/'+str(j)+'_tagvocab.json', 'w') as f:
        f.write(json.dumps(val))
        j+=1

args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# RUN
train(batches, args)
writer.close()

