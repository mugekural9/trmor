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

        self.num_dicts = args.num_dicts
        self.encoder_emb_dim = args.embedding_dim 
        self.rootdict_emb_dim = args.rootdict_emb_dim
        self.rootdict_emb_num = args.rootdict_emb_num
        self.orddict_emb_num  = args.orddict_emb_num

        self.vq_layer_root   = args.pretrained_model.vq_layer_root
        self.linear_suffix   = args.pretrained_model.linear_suffix
        self.vq_layer_suffix = args.pretrained_model.vq_layer_suffix
        self.ord_linears     = args.pretrained_model.ord_linears
        self.ord_vq_layers   = args.pretrained_model.ord_vq_layers


        self.linear = nn.Linear(512, len(vocab.word2id), bias=True)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
   
    def forward(self, surf):
         # fhs: (B,1,hdim)
        fhs, fcs, z = self.encoder(surf)
        vq_vectors = []; vq_inds = []
        quantized_input_root, vq_loss, quantized_inds = self.vq_layer_root(fhs,0)
        vq_vectors.append(quantized_input_root)
        vq_inds.append(quantized_inds)
        
        _fhs =  self.linear_suffix(fhs)
        quantized_input_suffix, vq_loss, quantized_inds = self.vq_layer_suffix(_fhs,0)
        vq_vectors.append(quantized_input_suffix)
        vq_inds.append(quantized_inds)

        # quantize thru ord dicts
        for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
            _fhs =  linear(fhs)
            # quantized_input: (B, 1, orddict_emb_dim)
            quantized_input, vq_loss, quantized_inds = vq_layer(_fhs,0)
            vq_vectors.append(quantized_input)
            vq_inds.append(quantized_inds)
        
        suffix_vectors = vq_vectors[1:]
        vq_vectors = (vq_vectors[0], torch.cat(suffix_vectors,dim=2)) 
        root_vector   = vq_vectors[0]
        output_logits = self.linear(fhs) 
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
