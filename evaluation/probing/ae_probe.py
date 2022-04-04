# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Linear probe
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class AE_Probe(nn.Module):
    def __init__(self, args, vocab, model_init, emb_init):
        super(AE_Probe, self).__init__()
        self.vocab = vocab
        self.device = args.device
        self.encoder =  args.pretrained_model.encoder
        self.linear = nn.Linear(args.nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
   
    def forward(self, surf, plot=False, ratiodict=None, last_iter=False):
        _, fhs = self.encoder(surf)
        # (batchsize, 1, vocab_size)
        sft = nn.Softmax(dim=2)
        output_logits = self.linear(fhs) 
        pred_tokens = torch.argmax(sft(output_logits),2)

        if plot:
            for j in range(len(pred_tokens)):
                w = self.linear.weight[pred_tokens[j].item()]
                prod =(fhs[j][0] * w)
                for i in range(512):
                    ratiodict[i] += prod[i].item()
        if last_iter:
            ratios=nn.functional.normalize(torch.tensor(ratiodict).unsqueeze(0)).squeeze(0)
            plt.plot(ratios.detach().cpu())
            plt.savefig('here')
        return output_logits

    def probe_loss(self, surf, y, plot=False, ratiodict=None,last_iter=False):
        output_logits = self(surf, plot, ratiodict, last_iter) 
        loss = self.loss(output_logits.squeeze(1), y.squeeze(1))
        acc,pred_tokens  = self.accuracy(output_logits, y)
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