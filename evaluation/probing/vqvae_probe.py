# -----------------------------------------------------------
# Date:        2022/03/17 
# Author:      Muge Kural
# Description: Linear probe
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np

class VQVAE_Probe(nn.Module):
    def __init__(self, args, vocab, model_init, emb_init):
        super(VQVAE_Probe, self).__init__()
        self.vocab    = vocab
        self.device   = args.device
        self.encoder  = args.pretrained_model.encoder
    
        self.linear_root   = args.pretrained_model.linear_root
        self.vq_layer_root = args.pretrained_model.vq_layer_root
        self.ord_linears   = args.pretrained_model.ord_linears
        self.ord_vq_layers = args.pretrained_model.ord_vq_layers

        self.linear = nn.Linear(512, len(vocab.word2id), bias=True)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
   
    def forward(self, surf):
         # fhs: (B,1,hdim)
        fhs, fcs, z = self.encoder(surf)
        _fhs = self.linear_root(fhs)
        orddicts = []
        rootdict, vq_loss, quantized_inds = self.vq_layer_root(_fhs,0)
        for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
            _fhs =  linear(fhs)
            quantized_input, _, _ = vq_layer(_fhs,0)
            orddicts.append(quantized_input)
        # (batchsize, 1, vocab_size)
        # concat: (b,1, 512)
        # rootdict: (b,1, 320)
        # orddicts[1 to end]: (b,1, 32)
        # orddict concat: (b,1, 192)
        output_logits = self.linear(fhs) 
        #output_logits = self.linear(rootdict) 
        #output_logits = self.linear(orddicts[3]) 
        #orddict_concat = torch.cat(orddicts,dim=2)
        #output_logits = self.linear(orddict_concat)
        #output_logits = self.linear(torch.cat((rootdict, orddict_concat),dim=2)) 
        
        return output_logits

    def probe_loss(self, surf, y, plot=False, ratiodict=None,last_iter=False):
        output_logits = self(surf) 
        loss = self.loss(output_logits.squeeze(1), y.squeeze(1))
        acc, pred_tokens  = self.accuracy(output_logits, y)
        # loss: avg over instances
        return loss.mean(), (acc,pred_tokens)

    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        return acc, pred_tokens
