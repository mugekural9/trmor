# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Linear probe
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np

class VAE_Probe(nn.Module):
    def __init__(self, args, vocab, model_init, emb_init):
        super(VAE_Probe, self).__init__()
        self.vocab = vocab
        self.device = args.device
        self.encoder =  args.pretrained_model.encoder
        self.linear = nn.Linear(args.nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
   
    def forward(self, surf):
        mu, logvar, fhs = self.encoder(surf)
        # (batchsize,1, nz)
        mu = mu.unsqueeze(1)
        # (batchsize, 1, vocab_size)
        output_logits = self.linear(fhs) 
        return output_logits

    def probe_loss(self, surf, y):
        output_logits = self(surf) 
        loss = self.loss(output_logits.squeeze(1), y.squeeze(1))
        acc  = self.accuracy(output_logits, y)
        # loss: avg over instances
        return loss.mean(), acc

    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        return acc
