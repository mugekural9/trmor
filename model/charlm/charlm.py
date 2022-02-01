# -----------------------------------------------------------
# Date:        2021/12/17 
# Author:      Muge Kural
# Description: Character-based LSTM language model
# -----------------------------------------------------------
import math
import torch
import torch.nn as nn

class CharLM(nn.Module):
    """docstring for CharLM"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(CharLM, self).__init__()
        self.ni = args.ni
        self.nh = args.nh
        self.vocab = vocab

        self.embed = nn.Embedding(len(vocab.word2id), args.ni, padding_idx=0)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)
        self.dropout_out = nn.Dropout(args.enc_dropout_out)

        # prediction layer
        self.pred_linear = nn.Linear(args.nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)


    def charlm_input(self, x):
        return x[:, :-1]

    def charlm_output(self, x):
        return x[:, 1:]

    def charlm_loss(self, x, recon_type='avg'):
        # (batch_size, seq_len)
        src = self.charlm_input(x)
        batch_size, seq_len = src.size()
        output_logits = self(src)
        
        # (batch_size, seq_len)
        tgt = self.charlm_output(x)
        # (batch_size * seq_len)
        _tgt = tgt.contiguous().view(-1)

        # (batch_size * seq_len)
        loss = self.loss(output_logits,  _tgt)

        # (batch_size, nsample)
        if recon_type=='avg':
            # avg over tokens
            loss = loss.mean()
        elif recon_type=='sum':
            # sum over tokens
            loss = loss.sum()
        elif recon_type == 'eos':
            # only eos token
            loss = loss[-1]

        return loss

    def forward(self, src):
        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(src)
        word_embed = self.dropout_in(word_embed)
        output, (last_state, last_cell) = self.lstm(word_embed)
        output = self.dropout_out(output)
        
        # (batch_size, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        
        # (batch_size * seq_len, vocab_size)
        return output_logits.view(-1, output_logits.size(2))


    def log_probability(self, x, recon_type='avg'):
        return -self.charlm_loss(x, recon_type)

