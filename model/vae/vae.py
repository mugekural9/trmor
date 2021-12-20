# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Character-based Variational Autoencoder 
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np

class VAE_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(VAE_Encoder, self).__init__()
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
            self.linear = nn.Linear(args.enc_nh*2, 2*args.nz, bias=False)
        else:
            self.linear = nn.Linear(args.enc_nh,  2*args.nz, bias=False)

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
        mean, logvar = self.linear(last_state).chunk(2, -1)
        return mean.squeeze(0), logvar.squeeze(0)
       
class VAE_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(VAE_Decoder, self).__init__()
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

class VAE(nn.Module):
    def __init__(self, args, vocab, model_init, emb_init):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(args, vocab, model_init, emb_init)
        self.decoder = VAE_Decoder(args, vocab, model_init, emb_init)

        self.args = args
        self.nz = args.nz

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def vae_loss(self, x, kl_weight):
        #remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        mu, logvar = self.encoder(x)
        # z: (batchsize, 1, nz)
        z = self.reparameterize(mu, logvar)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        # (batch_size)
        kl_loss = kl_weight * KL
        # (batch_size, seq_len, vocab_size)
        output_logits = self.decoder(src, z)
        # (batch_size * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))
        _tgt = tgt.contiguous().view(-1)
        # (batch_size * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        acc  = self.accuracy(output_logits, tgt)
        # recon_loss: avg over tokens
        recon_loss = recon_loss.mean()
        # kl_loss: avg over batches
        kl_loss = kl_loss.mean()
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss, acc

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

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
