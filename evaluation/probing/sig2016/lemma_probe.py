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

    def probe_loss(self,  hidden_state, y, lemma_vocab, surf_vocab, surf):
        output_logits = self(hidden_state) 
        loss = self.loss(output_logits.squeeze(1), y.squeeze(1))
        acc,pred_tokens, pred_words  = self.accuracy(output_logits, y, lemma_vocab, surf_vocab, surf)
        # loss: avg over instances
        return loss.mean(), (acc,pred_tokens, pred_words)

    def accuracy(self, output_logits, tgt, lemma_vocab, surf_vocab, surf):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        true_preds = []
        false_preds = []
        for i in range(pred_tokens.size(0)):
            gold_form = lemma_vocab.id2word(tgt[i].item())
            pred_form = lemma_vocab.id2word(pred_tokens[i].item())
            if pred_tokens[i] == tgt[i]:
                true_preds.append(''.join(surf_vocab.decode_sentence(surf[i])) + '\t' + gold_form+'\t'+pred_form)
            else:
                false_preds.append(''.join(surf_vocab.decode_sentence(surf[i])) + '\t' + gold_form+'\t'+pred_form)
        return acc, pred_tokens, (true_preds, false_preds)