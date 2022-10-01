# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Trainer of character-based variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

from bdb import Breakpoint
import sys, argparse, random, torch, json, matplotlib, os, math
import matplotlib.pyplot as plt
import numpy as np
from msved import MSVED
from common.utils import *
from torch import optim
from data.data_2 import build_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, lxtgt_ordered_batches, mode, args, kl_weight, tmp, log_shared_task=False):
    labeled_msved_numwords = args.valsize if mode =='val'  else args.tstsize
    indices = list(range(len(batches)))
    epoch_labeled_msved_loss = 0 
    epoch_labeled_msved_num_tokens = 0
    epoch_labeled_msved_num_tags = 0
    epoch_labeled_msved_recon_acc = 0
    epoch_labeled_msved_recon_loss = 0
    epoch_labeled_msved_kl_loss = 0
    epoch_labeled_msved_tag_pred_loss = 0
    epoch_labeled_msved_tag_acc = 0
    for i, idx in enumerate(indices):
        idx= indices[i]
        lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = batches[idx] 
        # Labeled MSVED 
        loss_labeled_msved, labeled_msved_tag_pred_loss, labeled_msved_tag_correct, labeled_msved_tag_total, labeled_msved_recon_loss, labeled_msved_kl_loss, labeled_msved_recon_acc = args.model.loss_labeled_msved(lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt, kl_weight, tmp, mode='test')
        epoch_labeled_msved_loss += loss_labeled_msved.sum().item()
        epoch_labeled_msved_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
        epoch_labeled_msved_recon_acc += labeled_msved_recon_acc
        epoch_labeled_msved_recon_loss += labeled_msved_recon_loss.sum().item()
        epoch_labeled_msved_kl_loss += labeled_msved_kl_loss.sum().item()
        epoch_labeled_msved_num_tags +=  labeled_msved_tag_total
        epoch_labeled_msved_tag_pred_loss += labeled_msved_tag_pred_loss.sum().item()
        epoch_labeled_msved_tag_acc += labeled_msved_tag_correct
    
    labeled_msved_loss = epoch_labeled_msved_loss / labeled_msved_numwords  
    labeled_msved_kl_loss = epoch_labeled_msved_kl_loss / labeled_msved_numwords
    labeled_msved_recon_loss = epoch_labeled_msved_recon_loss / epoch_labeled_msved_num_tokens
    labeled_msved_recon_acc = epoch_labeled_msved_recon_acc / epoch_labeled_msved_num_tokens
    labeled_msved_tag_pred_loss = epoch_labeled_msved_tag_pred_loss / epoch_labeled_msved_num_tags
    labeled_msved_tag_acc = epoch_labeled_msved_tag_acc / epoch_labeled_msved_num_tags
    args.logger.write('\nval--- labeled_msved_loss: %.4f,  labeled_msved_tag_pred_loss: %.4f, labeled_msved_tag_acc: %.4f, labeled_msved_kl_loss: %.4f,  labeled_msved_recon_loss: %.4f,  labeled_msved_recon_acc: %.4f'  % ( labeled_msved_loss,  labeled_msved_tag_pred_loss, labeled_msved_tag_acc, labeled_msved_kl_loss,  labeled_msved_recon_loss,  labeled_msved_recon_acc))

    if log_shared_task:
        lxtgt_to_lxsrc_msved_numwords = args.valsize if mode =='val'  else args.tstsize
        ltgtindices = list(range(len(lxtgt_ordered_batches)))
        epoch_lxtgt_to_lxsrc_msved_loss = 0 
        epoch_lxtgt_to_lxsrc_msved_num_tokens = 0
        epoch_lxtgt_to_lxsrc_msved_recon_acc = 0
        epoch_lxtgt_to_lxsrc_msved_recon_loss = 0
        epoch_lxtgt_to_lxsrc_msved_kl_loss = 0
        true = 0
        with open('semisup_tgt_to_src_reinflections.txt','w') as writer:
            for i, idx in enumerate(ltgtindices):
                lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = lxtgt_ordered_batches[idx] 
                # MSVED from lxtgt to lxsrc
                loss_lxtgt_to_lxsrc_msved, lxtgt_to_lxsrc_msved_recon_loss, lxtgt_to_lxsrc_msved_kl_loss, lxtgt_to_lxsrc_msved_recon_acc, _ = args.model.loss_lxtgt_to_lxsrc_msved(lxsrc, lxtgt, kl_weight, tmp)
                gold_str =  ''.join(args.model.decoder.vocab.decode_sentence(lxsrc.squeeze(0)[1:]))
                pred_str =  ''.join(args.model.decoder.vocab.decode_sentence(_.squeeze(0)))
                if gold_str == pred_str:
                    true+=1
                writer.write(gold_str+'\t'+pred_str+'\n')
                epoch_lxtgt_to_lxsrc_msved_loss += loss_lxtgt_to_lxsrc_msved.sum().item()
                epoch_lxtgt_to_lxsrc_msved_num_tokens +=  torch.sum(lxsrc[:,1:] !=0).item()
                epoch_lxtgt_to_lxsrc_msved_recon_acc += lxtgt_to_lxsrc_msved_recon_acc
                epoch_lxtgt_to_lxsrc_msved_recon_loss += lxtgt_to_lxsrc_msved_recon_loss.sum().item()
                epoch_lxtgt_to_lxsrc_msved_kl_loss += lxtgt_to_lxsrc_msved_kl_loss.sum().item()
        lxtgt_to_lxsrc_msved_loss = epoch_lxtgt_to_lxsrc_msved_loss / lxtgt_to_lxsrc_msved_numwords  
        lxtgt_to_lxsrc_msved_kl_loss = epoch_lxtgt_to_lxsrc_msved_kl_loss / lxtgt_to_lxsrc_msved_numwords
        lxtgt_to_lxsrc_msved_recon_loss = epoch_lxtgt_to_lxsrc_msved_recon_loss / epoch_lxtgt_to_lxsrc_msved_num_tokens
        lxtgt_to_lxsrc_msved_recon_acc = epoch_lxtgt_to_lxsrc_msved_recon_acc / epoch_lxtgt_to_lxsrc_msved_num_tokens
        reinflect_acc = (true/len(ltgtindices))
        args.logger.write('\nval--- lxtgt_to_lxsrc_msved_loss: %.4f,  lxtgt_to_lxsrc_msved_kl_loss: %.4f,  lxtgt_to_lxsrc_msved_recon_loss: %.4f,  lxtgt_to_lxsrc_msved_recon_acc: %.4f'  % ( lxtgt_to_lxsrc_msved_loss,  lxtgt_to_lxsrc_msved_kl_loss,  lxtgt_to_lxsrc_msved_recon_loss,  lxtgt_to_lxsrc_msved_recon_acc))
        args.logger.write('\nreinflect acc: %.4f' % reinflect_acc)
    return labeled_msved_loss


def train(data, args):
    lxsrc_ordered_batches, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, valbatches, tstbatches, ubatches = data
    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numlxsrcbatches = len(lxsrc_ordered_batches); lsrcindices = list(range(numlxsrcbatches))
    numlxtgtbatches = len(lxtgt_ordered_batches); ltgtindices = list(range(numlxtgtbatches))

    numubatches = len(ubatches); uindices = list(range(numubatches))

    ux_msvae_numwords = args.usize
    lxsrc_msvae_numwords = args.trnsize
    lxtgt_to_lxsrc_msved_numwords = args.trnsize
    labeled_msved_numwords = args.trnsize
    lxtgt_labeled_msvae_numwords = args.trnsize

    best_loss = 1e4
    tmp=1.0
    update_ind =0

    for epc in range(args.epochs):
        epoch_ux_msvae_loss = 0 
        epoch_ux_msvae_num_tokens = 0
        epoch_ux_msvae_recon_acc = 0
        epoch_ux_msvae_recon_loss = 0
        epoch_ux_msvae_kl_loss = 0

        epoch_lxsrc_msvae_loss = 0 
        epoch_lxsrc_msvae_num_tokens = 0
        epoch_lxsrc_msvae_recon_acc = 0
        epoch_lxsrc_msvae_recon_loss = 0
        epoch_lxsrc_msvae_kl_loss = 0

        epoch_lxtgt_labeled_msvae_loss = 0 
        epoch_lxtgt_labeled_msvae_num_tokens = 0
        epoch_lxtgt_labeled_msvae_num_tags = 0
        epoch_lxtgt_labeled_msvae_recon_acc = 0
        epoch_lxtgt_labeled_msvae_recon_loss = 0
        epoch_lxtgt_labeled_msvae_kl_loss = 0
        epoch_lxtgt_labeled_msvae_tag_pred_loss = 0
        epoch_lxtgt_labeled_msvae_tag_acc = 0

        epoch_lxtgt_to_lxsrc_msved_loss = 0 
        epoch_lxtgt_to_lxsrc_msved_num_tokens = 0
        epoch_lxtgt_to_lxsrc_msved_recon_acc = 0
        epoch_lxtgt_to_lxsrc_msved_recon_loss = 0
        epoch_lxtgt_to_lxsrc_msved_kl_loss = 0

        epoch_labeled_msved_loss = 0 
        epoch_labeled_msved_num_tokens = 0
        epoch_labeled_msved_num_tags = 0
        epoch_labeled_msved_recon_acc = 0
        epoch_labeled_msved_recon_loss = 0
        epoch_labeled_msved_kl_loss = 0
        epoch_labeled_msved_tag_pred_loss = 0
        epoch_labeled_msved_tag_acc = 0

        random.shuffle(lsrcindices) # this breaks continuity if there is any
        random.shuffle(ltgtindices) # this breaks continuity if there is any
        random.shuffle(uindices) # this breaks continuity if there is any


        for i, uidx in enumerate(uindices):
            loss = torch.tensor(0.0).to('cuda')
            if update_ind % args.update_temp == 0:
                tmp = get_temp(update_ind)
            kl_weight = get_kl_weight(update_ind, 0.2, 150000.0)
            args.model.zero_grad()
            
            ux = ubatches[uidx] 
            update_ind +=1
            batch_loss = torch.tensor(0.0).to('cuda')

            if i < len(lxsrc_ordered_batches):
                lidx= lsrcindices[i]
                lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = lxsrc_ordered_batches[lidx] 
                # MSVAE with lxsrc
                loss_lxsrc_msvae, lxsrc_msvae_recon_loss, lxsrc_msvae_kl_loss, lxsrc_msvae_recon_acc, _ = args.model.loss_lxsrc_msvae(lxsrc, kl_weight, tmp)
                lxsrc_msvae_batch_loss = loss_lxsrc_msvae.mean()
                batch_loss += lxsrc_msvae_batch_loss
                epoch_lxsrc_msvae_loss += loss_lxsrc_msvae.sum().item()
                epoch_lxsrc_msvae_num_tokens +=  torch.sum(lxsrc[:,1:] !=0).item()
                epoch_lxsrc_msvae_recon_acc += lxsrc_msvae_recon_acc
                epoch_lxsrc_msvae_recon_loss += lxsrc_msvae_recon_loss.sum().item()
                epoch_lxsrc_msvae_kl_loss += lxsrc_msvae_kl_loss.sum().item()
                # Labeled MSVED 
                loss_labeled_msved, labeled_msved_tag_pred_loss, labeled_msved_tag_correct, labeled_msved_tag_total, labeled_msved_recon_loss, labeled_msved_kl_loss, labeled_msved_recon_acc = args.model.loss_labeled_msved(lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt, kl_weight, tmp)
                labeled_msved_batch_loss = loss_labeled_msved.mean()
                batch_loss += labeled_msved_batch_loss
                epoch_labeled_msved_loss += loss_labeled_msved.sum().item()
                epoch_labeled_msved_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
                epoch_labeled_msved_recon_acc += labeled_msved_recon_acc
                epoch_labeled_msved_recon_loss += labeled_msved_recon_loss.sum().item()
                epoch_labeled_msved_kl_loss += labeled_msved_kl_loss.sum().item()
                epoch_labeled_msved_num_tags +=  labeled_msved_tag_total
                epoch_labeled_msved_tag_pred_loss += labeled_msved_tag_pred_loss.sum().item()
                epoch_labeled_msved_tag_acc += labeled_msved_tag_correct
            else:
                random.shuffle(lsrcindices) # this breaks continuity if there is any
                lidx= lsrcindices[0]
                lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = lxsrc_ordered_batches[lidx] 
                # MSVAE with lxsrc
                loss_lxsrc_msvae, lxsrc_msvae_recon_loss, lxsrc_msvae_kl_loss, lxsrc_msvae_recon_acc, _ = args.model.loss_lxsrc_msvae(lxsrc, kl_weight, tmp)
                lxsrc_msvae_batch_loss = loss_lxsrc_msvae.mean()
                batch_loss += lxsrc_msvae_batch_loss
                epoch_lxsrc_msvae_loss += loss_lxsrc_msvae.sum().item()
                epoch_lxsrc_msvae_num_tokens +=  torch.sum(lxsrc[:,1:] !=0).item()
                epoch_lxsrc_msvae_recon_acc += lxsrc_msvae_recon_acc
                epoch_lxsrc_msvae_recon_loss += lxsrc_msvae_recon_loss.sum().item()
                epoch_lxsrc_msvae_kl_loss += lxsrc_msvae_kl_loss.sum().item()
                # Labeled MSVED 
                loss_labeled_msved, labeled_msved_tag_pred_loss, labeled_msved_tag_correct, labeled_msved_tag_total, labeled_msved_recon_loss, labeled_msved_kl_loss, labeled_msved_recon_acc = args.model.loss_labeled_msved(lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt, kl_weight, tmp)
                labeled_msved_batch_loss = loss_labeled_msved.mean()
                batch_loss += labeled_msved_batch_loss
                epoch_labeled_msved_loss += loss_labeled_msved.sum().item()
                epoch_labeled_msved_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
                epoch_labeled_msved_recon_acc += labeled_msved_recon_acc
                epoch_labeled_msved_recon_loss += labeled_msved_recon_loss.sum().item()
                epoch_labeled_msved_kl_loss += labeled_msved_kl_loss.sum().item()
                epoch_labeled_msved_num_tags +=  labeled_msved_tag_total
                epoch_labeled_msved_tag_pred_loss += labeled_msved_tag_pred_loss.sum().item()
                epoch_labeled_msved_tag_acc += labeled_msved_tag_correct
            
            if i < len(lxtgt_ordered_batches):
                lidx= ltgtindices[i]
                lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = lxtgt_ordered_batches[lidx] 
                # Labeled MSVAE with lxtgt
                loss_lxtgt_labeled_msvae, lxtgt_labeled_msvae_tag_pred_loss, lxtgt_labeled_msvae_tag_correct, lxtgt_labeled_msvae_tag_total, lxtgt_labeled_msvae_recon_loss, lxtgt_labeled_msvae_kl_loss, lxtgt_labeled_msvae_recon_acc = args.model.loss_lxtgt_labeled_msvae(lxtgt, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, kl_weight, tmp)
                lxtgt_labeled_msvae_batch_loss = loss_lxtgt_labeled_msvae.mean()
                batch_loss += lxtgt_labeled_msvae_batch_loss
                epoch_lxtgt_labeled_msvae_loss += loss_lxtgt_labeled_msvae.sum().item()
                epoch_lxtgt_labeled_msvae_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
                epoch_lxtgt_labeled_msvae_recon_acc += lxtgt_labeled_msvae_recon_acc
                epoch_lxtgt_labeled_msvae_recon_loss += lxtgt_labeled_msvae_recon_loss.sum().item()
                epoch_lxtgt_labeled_msvae_kl_loss += lxtgt_labeled_msvae_kl_loss.sum().item()
                epoch_lxtgt_labeled_msvae_num_tags +=  lxtgt_labeled_msvae_tag_total
                epoch_lxtgt_labeled_msvae_tag_pred_loss += lxtgt_labeled_msvae_tag_pred_loss.sum().item()
                epoch_lxtgt_labeled_msvae_tag_acc += lxtgt_labeled_msvae_tag_correct
                # MSVED from lxtgt to lxsrc
                loss_lxtgt_to_lxsrc_msved, lxtgt_to_lxsrc_msved_recon_loss, lxtgt_to_lxsrc_msved_kl_loss, lxtgt_to_lxsrc_msved_recon_acc,_ = args.model.loss_lxtgt_to_lxsrc_msved(lxsrc, lxtgt, kl_weight, tmp)
                lxtgt_to_lxsrc_msved_batch_loss = loss_lxtgt_to_lxsrc_msved.mean()
                batch_loss += lxtgt_to_lxsrc_msved_batch_loss
                epoch_lxtgt_to_lxsrc_msved_loss += loss_lxtgt_to_lxsrc_msved.sum().item()
                epoch_lxtgt_to_lxsrc_msved_num_tokens +=  torch.sum(lxsrc[:,1:] !=0).item()
                epoch_lxtgt_to_lxsrc_msved_recon_acc += lxtgt_to_lxsrc_msved_recon_acc
                epoch_lxtgt_to_lxsrc_msved_recon_loss += lxtgt_to_lxsrc_msved_recon_loss.sum().item()
                epoch_lxtgt_to_lxsrc_msved_kl_loss += lxtgt_to_lxsrc_msved_kl_loss.sum().item()
            else:
                random.shuffle(ltgtindices) # this breaks continuity if there is any
                lidx= ltgtindices[0]
                lxsrc, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lxtgt  = lxtgt_ordered_batches[lidx] 
                # Labeled MSVAE with lxtgt
                loss_lxtgt_labeled_msvae, lxtgt_labeled_msvae_tag_pred_loss, lxtgt_labeled_msvae_tag_correct, lxtgt_labeled_msvae_tag_total, lxtgt_labeled_msvae_recon_loss, lxtgt_labeled_msvae_kl_loss, lxtgt_labeled_msvae_recon_acc = args.model.loss_lxtgt_labeled_msvae(lxtgt, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, kl_weight, tmp)
                lxtgt_labeled_msvae_batch_loss = loss_lxtgt_labeled_msvae.mean()
                batch_loss += lxtgt_labeled_msvae_batch_loss
                epoch_lxtgt_labeled_msvae_loss += loss_lxtgt_labeled_msvae.sum().item()
                epoch_lxtgt_labeled_msvae_num_tokens +=  torch.sum(lxtgt[:,1:] !=0).item()
                epoch_lxtgt_labeled_msvae_recon_acc += lxtgt_labeled_msvae_recon_acc
                epoch_lxtgt_labeled_msvae_recon_loss += lxtgt_labeled_msvae_recon_loss.sum().item()
                epoch_lxtgt_labeled_msvae_kl_loss += lxtgt_labeled_msvae_kl_loss.sum().item()
                epoch_lxtgt_labeled_msvae_num_tags +=  lxtgt_labeled_msvae_tag_total
                epoch_lxtgt_labeled_msvae_tag_pred_loss += lxtgt_labeled_msvae_tag_pred_loss.sum().item()
                epoch_lxtgt_labeled_msvae_tag_acc += lxtgt_labeled_msvae_tag_correct
                # MSVED from lxtgt to lxsrc
                loss_lxtgt_to_lxsrc_msved, lxtgt_to_lxsrc_msved_recon_loss, lxtgt_to_lxsrc_msved_kl_loss, lxtgt_to_lxsrc_msved_recon_acc, _ = args.model.loss_lxtgt_to_lxsrc_msved(lxsrc, lxtgt, kl_weight, tmp)
                lxtgt_to_lxsrc_msved_batch_loss = loss_lxtgt_to_lxsrc_msved.mean()
                batch_loss += lxtgt_to_lxsrc_msved_batch_loss
                epoch_lxtgt_to_lxsrc_msved_loss += loss_lxtgt_to_lxsrc_msved.sum().item()
                epoch_lxtgt_to_lxsrc_msved_num_tokens +=  torch.sum(lxsrc[:,1:] !=0).item()
                epoch_lxtgt_to_lxsrc_msved_recon_acc += lxtgt_to_lxsrc_msved_recon_acc
                epoch_lxtgt_to_lxsrc_msved_recon_loss += lxtgt_to_lxsrc_msved_recon_loss.sum().item()
                epoch_lxtgt_to_lxsrc_msved_kl_loss += lxtgt_to_lxsrc_msved_kl_loss.sum().item()
                
            
            # MSVAE with ux
            loss_ux_msvae, ux_msvae_recon_loss, ux_msvae_kl_loss, ux_msvae_recon_acc, _ = args.model.loss_ux_msvae(ux, kl_weight, tmp)
            ux_msvae_batch_loss = loss_ux_msvae.mean()
          
            epoch_ux_msvae_loss += loss_ux_msvae.sum().item()
            epoch_ux_msvae_num_tokens +=  torch.sum(ux[:,1:] !=0).item()
            epoch_ux_msvae_recon_acc += ux_msvae_recon_acc
            epoch_ux_msvae_recon_loss += ux_msvae_recon_loss.sum().item()
            epoch_ux_msvae_kl_loss += ux_msvae_kl_loss.sum().item()

            batch_loss += 0.8 * ux_msvae_batch_loss
            batch_loss.backward()
            opt.step()
     
        ux_msvae_loss = epoch_ux_msvae_loss / ux_msvae_numwords  
        ux_msvae_kl_loss = epoch_ux_msvae_kl_loss / ux_msvae_numwords
        ux_msvae_recon_loss = epoch_ux_msvae_recon_loss / epoch_ux_msvae_num_tokens
        ux_msvae_recon_acc = epoch_ux_msvae_recon_acc / epoch_ux_msvae_num_tokens

        lxsrc_msvae_loss = epoch_lxsrc_msvae_loss / lxsrc_msvae_numwords  
        lxsrc_msvae_kl_loss = epoch_lxsrc_msvae_kl_loss / lxsrc_msvae_numwords
        lxsrc_msvae_recon_loss = epoch_lxsrc_msvae_recon_loss / epoch_lxsrc_msvae_num_tokens
        lxsrc_msvae_recon_acc = epoch_lxsrc_msvae_recon_acc / epoch_lxsrc_msvae_num_tokens

        lxtgt_labeled_msvae_loss = epoch_lxtgt_labeled_msvae_loss / lxtgt_labeled_msvae_numwords  
        lxtgt_labeled_msvae_kl_loss = epoch_lxtgt_labeled_msvae_kl_loss / lxtgt_labeled_msvae_numwords
        lxtgt_labeled_msvae_recon_loss = epoch_lxtgt_labeled_msvae_recon_loss / epoch_lxtgt_labeled_msvae_num_tokens
        lxtgt_labeled_msvae_recon_acc = epoch_lxtgt_labeled_msvae_recon_acc / epoch_lxtgt_labeled_msvae_num_tokens
        lxtgt_labeled_msvae_tag_pred_loss = epoch_lxtgt_labeled_msvae_tag_pred_loss / epoch_lxtgt_labeled_msvae_num_tags
        lxtgt_labeled_msvae_tag_acc = epoch_lxtgt_labeled_msvae_tag_acc / epoch_lxtgt_labeled_msvae_num_tags
  
        lxtgt_to_lxsrc_msved_loss = epoch_lxtgt_to_lxsrc_msved_loss / lxtgt_to_lxsrc_msved_numwords  
        lxtgt_to_lxsrc_msved_kl_loss = epoch_lxtgt_to_lxsrc_msved_kl_loss / lxtgt_to_lxsrc_msved_numwords
        lxtgt_to_lxsrc_msved_recon_loss = epoch_lxtgt_to_lxsrc_msved_recon_loss / epoch_lxtgt_to_lxsrc_msved_num_tokens
        lxtgt_to_lxsrc_msved_recon_acc = epoch_lxtgt_to_lxsrc_msved_recon_acc / epoch_lxtgt_to_lxsrc_msved_num_tokens

        labeled_msved_loss = epoch_labeled_msved_loss / labeled_msved_numwords  
        labeled_msved_kl_loss = epoch_labeled_msved_kl_loss / labeled_msved_numwords
        labeled_msved_recon_loss = epoch_labeled_msved_recon_loss / epoch_labeled_msved_num_tokens
        labeled_msved_recon_acc = epoch_labeled_msved_recon_acc / epoch_labeled_msved_num_tokens
        labeled_msved_tag_pred_loss = epoch_labeled_msved_tag_pred_loss / epoch_labeled_msved_num_tags
        labeled_msved_tag_acc = epoch_labeled_msved_tag_acc / epoch_labeled_msved_num_tags


        args.logger.write('\nepoch: %.1d, kl_weight: %.2f, tmp: %.2f' % (epc, kl_weight, tmp))
        #args.logger.write('\ntrn--- ux_msvae_loss: %.4f,  ux_msvae_kl_loss: %.4f,  ux_msvae_recon_loss: %.4f,  ux_msvae_recon_acc: %.4f'  % ( ux_msvae_loss,  ux_msvae_kl_loss,  ux_msvae_recon_loss,  ux_msvae_recon_acc))
        #args.logger.write('\ntrn--- lxsrc_msvae_loss: %.4f,  lxsrc_msvae_kl_loss: %.4f,  lxsrc_msvae_recon_loss: %.4f,  lxsrc_msvae_recon_acc: %.4f'  % ( lxsrc_msvae_loss,  lxsrc_msvae_kl_loss,  lxsrc_msvae_recon_loss,  lxsrc_msvae_recon_acc))
        #args.logger.write('\ntrn--- lxtgt_labeled_msvae_loss: %.4f,  lxtgt_labeled_msvae_tag_pred_loss: %.4f, lxtgt_labeled_msvae_tag_acc: %.4f, lxtgt_labeled_msvae_kl_loss: %.4f,  lxtgt_labeled_msvae_recon_loss: %.4f,  lxtgt_labeled_msvae_recon_acc: %.4f'  % ( lxtgt_labeled_msvae_loss,  lxtgt_labeled_msvae_tag_pred_loss, lxtgt_labeled_msvae_tag_acc, lxtgt_labeled_msvae_kl_loss,  lxtgt_labeled_msvae_recon_loss,  lxtgt_labeled_msvae_recon_acc))
        args.logger.write('\ntrn--- lxtgt_to_lxsrc_msved_loss: %.4f,  lxtgt_to_lxsrc_msved_kl_loss: %.4f,  lxtgt_to_lxsrc_msved_recon_loss: %.4f,  lxtgt_to_lxsrc_msved_recon_acc: %.4f'  % ( lxtgt_to_lxsrc_msved_loss,  lxtgt_to_lxsrc_msved_kl_loss,  lxtgt_to_lxsrc_msved_recon_loss,  lxtgt_to_lxsrc_msved_recon_acc))
        #args.logger.write('\ntrn--- labeled_msved_loss: %.4f,  labeled_msved_tag_pred_loss: %.4f, labeled_msved_tag_acc: %.4f, labeled_msved_kl_loss: %.4f,  labeled_msved_recon_loss: %.4f,  labeled_msved_recon_acc: %.4f'  % ( labeled_msved_loss,  labeled_msved_tag_pred_loss, labeled_msved_tag_acc, labeled_msved_kl_loss,  labeled_msved_recon_loss,  labeled_msved_recon_acc))

        # VAL
        args.model.eval()
        with torch.no_grad():
            if epc % 10 == 0:
                loss = test(valbatches, lxtgt_ordered_batches_TST, "val", args, kl_weight, tmp, log_shared_task=True)
            else:
                loss = test(valbatches, lxtgt_ordered_batches, "val", args, kl_weight, tmp, log_shared_task=False)
            if loss < best_loss:
                args.logger.write('\nupdate best loss \n')
                best_loss = loss

        # SHARED TASK
        if epc %10==0 or epc>90:
            shared_task_gen(tstbatches, args, epc)
            #oracle_src_to_tgt(args, epc, kl_weight, tmp)
            #oracle_tgt_to_src(args, epc, kl_weight, tmp)
            torch.save(args.model.state_dict(), args.save_path+'_'+str(epc))

        args.model.train()
  

def get_temp(update_ind):
    return max(0.5, math.exp(-3 * 1e-5 * update_ind))


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


def shared_task_gen(batches, args, epc):
    indices = list(range( len(batches)))
    correct = 0
    with open('semisup_msved_shared_task_tst_beam.txt', 'w') as f:
        for i, idx in enumerate(indices):
            # (batchsize)
            surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, gold_reinflect_surf  = batches[idx] 
            inflected_form = ''.join(surf_vocab.decode_sentence(surf.squeeze(0)))
            reinflected_form = args.model.generate(surf,case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss)
            gold_reinflected_form = ''.join(surf_vocab.decode_sentence(gold_reinflect_surf.squeeze(0)))
            f.write(inflected_form+'\t'+reinflected_form+ '\t'+gold_reinflected_form+ '\n')
            #gold_reinflected_form = gold_reinflected_form[3:]
            if reinflected_form == gold_reinflected_form:
             correct +=1
    args.logger.write('\nTST SET ACCURACY: %.3f' % (correct/1600))


def oracle_src_to_tgt(args, epc, kl_weight, tmp):
    with open('Semisup_msved_src_to_tgt.txt', 'w') as f:
        with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
            # (batchsize)
            true = 0
            count = 0
            for line in reader:
                count+=1
                if count>1600:
                    break
                split_line = line.split('\t')
                asked_tag  = split_line[1]
                inflected_word = split_line[0]
                reinflected_word = split_line[2].strip()
                lxsrc = torch.tensor([args.model.decoder.vocab.word2id['<s>']] + args.model.decoder.vocab.encode_sentence(inflected_word) + [args.model.decoder.vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
                lxtgt = torch.tensor([args.model.decoder.vocab.word2id['<s>']] + args.model.decoder.vocab.encode_sentence(reinflected_word) + [args.model.decoder.vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
                breakpoint()
                
                # MSVED from lxsrc to lxtgt
                loss_lxsrc_to_lxtgt_msved, lxsrc_to_lxtgt_msved_recon_loss, lxsrc_to_lxtgt_msved_kl_loss, lxsrc_to_lxtgt_msved_recon_acc, recon_preds = args.model.loss_lxtgt_to_lxsrc_msved(lxtgt, lxsrc, kl_weight, tmp, mode='test')
                lxsrc_str = ''.join(args.model.decoder.vocab.decode_sentence(lxsrc.squeeze(0)[1:]))
                lxtgt_str = ''.join(args.model.decoder.vocab.decode_sentence(lxtgt.squeeze(0)[1:]))
                pred = ''.join(args.model.decoder.vocab.decode_sentence(recon_preds.squeeze(0)))
                f.write(lxsrc_str+'\t'+lxtgt_str+'\t'+pred+'\n')
                if pred == lxtgt_str:
                    true+=1
            args.logger.write('\nShared Task oracle from lxsrc to lxtgt: %.4f' % (true/count))



# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 100
args.opt= 'Adam'; args.lr = 0.001
args.task = 'msved'
args.seq_to_no_pad = 'surface'
# data
args.trndata  = 'data/sigmorphon2016/turkish-task3-train'
args.valdata  = 'data/sigmorphon2016/turkish-task3-test'
args.tstdata  = 'data/sigmorphon2016/turkish-task3-test'
args.unlabeled_data = 'data/sigmorphon2016/zhou_ux.txt'

args.update_temp = 2000

args.surface_vocab_file = args.trndata
args.maxtrnsize = 700000000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, surf_vocab, tag_vocabs = build_data(args)

trndata, vlddata, tstdata, udata = rawdata
args.trnsize , args.valsize, args.tstsize, args.usize = len(trndata), len(vlddata), len(trndata), len(udata)
# model
args.mname = 'msved' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 300; args.nz = 150; 
args.enc_nh = 256; args.dec_nh = 256
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.4; 
args.model = MSVED(args, surf_vocab, tag_vocabs, model_init, emb_init)
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
    f.write(json.dumps(surf_vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')
# plotting
args.fig, args.axs = plt.subplots(3)
args.plt_style = pstyle = '-'
args.fig.tight_layout() 

# RUN
#train(batches, args)
train(batches, args)
plt.savefig(args.fig_path)