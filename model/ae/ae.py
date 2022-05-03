# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Character-based Autoencoder 
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np

class AE_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(AE_Encoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz

        self.embed = nn.Embedding(len(vocab.word2id), args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)

        # dimension transformation to z
        if self.lstm.bidirectional:
            self.linear = nn.Linear(args.enc_nh*2, args.nz, bias=False)
        else:
            self.linear = nn.Linear(args.enc_nh,  args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)


    def forward(self, input):
        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        _, (last_state, last_cell) = self.lstm(word_embed)
        if self.lstm.bidirectional:
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)

        # (1, batch_size, args.nz)
        z = self.linear(last_state)
        # (batch_size, 1, args.nz)
        z = z.permute(1,0,2)
        last_state = last_state.permute(1,0,2)
        return z, last_state
       
class AE_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(AE_Decoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab.word2id), args.ni, padding_idx=0)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

        # concatenate z with input
        self.lstm = nn.LSTM(input_size=args.ni + args.nz,
                            hidden_size=args.dec_nh,
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(args.dec_nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input, z):
        batch_size, _, _ = z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)
        z_ = z.expand(batch_size, seq_len, self.nz)
        # (batch_size, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)
        # (batch_size, nz)
        z = z.view(batch_size, self.nz)
        # (1, batch_size, dec_nh)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))
        output = self.dropout_out(output)
        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits

class AE(nn.Module):
    def __init__(self, args, vocab, model_init, emb_init):
        super(AE, self).__init__()
        self.encoder = AE_Encoder(args, vocab, model_init, emb_init)
        self.decoder = AE_Decoder(args, vocab, model_init, emb_init)

    def loss(self, x):
        # z: (batchsize, 1, nz), encoder_fhs: (batchsize, 1, enc_nh)
        z, encoder_fhs = self.encoder(x)
        recon_loss, recon_acc = self.recon_loss(x, z, recon_type='sum')
        # avg over batches
        recon_loss = recon_loss
        return recon_loss,  recon_acc, encoder_fhs

    def recon_loss(self, x, z, recon_type='avg'):
        #remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decoder(src, z)
        # (batch_size * nsample * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))
        _tgt = tgt.contiguous().view(-1)
     
        # (batch_size * nsample * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, nsample, seq_len)
        recon_loss = recon_loss.view(batch_size, n_sample, -1)

        # (batch_size, nsample)
        if recon_type=='avg':
            # avg over tokens
            recon_loss = recon_loss.mean(-1)
        elif recon_type=='sum':
            # sum over tokens
            recon_loss = recon_loss.sum(-1)
        elif recon_type == 'eos':
            # only eos token
            recon_loss = recon_loss[:,:,-1]

        # avg over batches and samples
        recon_acc  = self.accuracy(output_logits, tgt)
        return recon_loss, recon_acc

    def log_probability(self, x, z, recon_type='avg'):
        return -self.recon_loss(x, z, recon_type)[0]

    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        return acc

    def log_preds():
        # log preds to make error analysis
        correct_predictions = []; wrong_predictions = []
        for i in range(len(src)):
            src_str   = ''.join(self.decoder.vocab.decode_sentence(src[i])[1:])
            tgt_str   = ''.join(self.decoder.vocab.decode_sentence(tgt[i]))
            pred_str  = ''.join(self.decoder.vocab.decode_sentence(pred_tokens[i]))
            if tgt_str == pred_str:
                correct_predictions.append('%s target: %s pred: %s' % (src_str, tgt_str, pred_str))
            else:
                wrong_predictions.append('%s target: %s pred: %s' % (src_str, tgt_str, pred_str))
        preds = (correct_predictions, wrong_predictions)
        return preds
