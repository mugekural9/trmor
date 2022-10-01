# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Linear probe
# -----------------------------------------------------------

import torch
import torch.nn as nn

class Probe(nn.Module):
    def __init__(self, args, vocabsize):
        super(Probe, self).__init__()
        self.device = args.device
        self.linear = nn.Linear(args.nh,  vocabsize, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=0, reduce=False)
   
    def forward(self, hidden_state):
        # (batchsize, 1, vocab_size)
        sft = nn.Softmax(dim=2)
        output_logits = self.linear(hidden_state) 
        return output_logits

    def probe_loss(self, hidden_state, y):
        output_logits = self(hidden_state) 
        loss = self.loss(output_logits.squeeze(1), y.squeeze(1))
        acc,pred_tokens  = self.accuracy(output_logits, y)
        # loss: avg over instances
        return loss.mean(), (acc,pred_tokens)

    def probe_random_loss(self, y):
        random_preds = torch.randint(1, self.linear.weight.size(0), y.size()).to('cuda')
        acc = (random_preds == y).sum().item()
        return 0, (acc,random_preds)


    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        return acc, pred_tokens