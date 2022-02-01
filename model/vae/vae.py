# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Character-based Variational Autoencoder 
# -----------------------------------------------------------

import math
import torch
import torch.nn as nn
import numpy as np
from common.utils import log_sum_exp

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
        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        return mean.squeeze(0), logvar.squeeze(0), last_state
       
    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            mu, logvar, _ = self.forward(x)
        else:
            mu, logvar, _ = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

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
        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        if n_sample == 1:
            z_ = z.expand(batch_size, seq_len, self.nz)
        else:
            word_embed = word_embed.unsqueeze(1).expand(batch_size, n_sample, seq_len, self.ni) \
                                   .contiguous()

            # (batch_size * n_sample, seq_len, ni)
            word_embed = word_embed.view(batch_size * n_sample, seq_len, self.ni)
            z_ = z.unsqueeze(2).expand(batch_size, n_sample, seq_len, self.nz).contiguous()
            z_ = z_.view(batch_size * n_sample, seq_len, self.nz)

        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)
        # (batch_size * n_sample, nz)
        z = z.view(batch_size * n_sample, self.nz)

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

    def loss(self, x, kl_weight):
        mu, logvar, _ = self.encoder(x)
        # (batchsize, 1, nz)
        z = self.reparameterize(mu, logvar)
        recon_loss, recon_acc = self.recon_loss(x, z, recon_type='sum')
        kl_loss = kl_weight * self.kl_loss(mu,logvar)
        # avg over batches
        recon_loss = recon_loss.mean()
        kl_loss = kl_loss.mean()
        loss = recon_loss + kl_loss
        return loss, recon_loss, kl_loss, recon_acc

    def kl_loss(self, mu, logvar):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return KL

    def recon_loss(self, x, z, recon_type='avg'):
        #remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decoder(src, z)

        if n_sample == 1:
            _tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            _tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)
        
        # (batch_size * nsample * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

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
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """
        return -self.recon_loss(x, z, recon_type)[0]

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

    def eval_cond_ll(self, x, z, recon_type='avg'):
        """compute log p(x|z)
        """
        return self.log_probability(x, z, recon_type)

    def eval_prior_dist(self, zrange):
        """perform grid search to calculate the true posterior
        Args:
            zrange: tensor
                different z points that will be evaluated, with
                shape (k^2, nz), where k=(zmax - zmin)/space
        """
        # (k^2)
        return self.prior.log_prob(zrange).sum(dim=-1)
    
    def eval_complete_ll(self, x, z, recon_type='avg'):
        """compute log p(z,x)
        Args:
            x: Tensor
                input with shape [batch, seq_len]
            z: Tensor
                evaluation points with shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log p(z,x) Tensor with shape [batch, nsamples]
        """
        # [batch, nsamples]
        log_prior = self.eval_prior_dist(z)
        log_gen = self.eval_cond_ll(x, z, recon_type)
        return log_prior + log_gen

    def eval_inference_dist(self, x, z, param=None):
        """
        Returns: Tensor
            Tensor: the posterior density tensor with
                shape (batch_size, nsamples)
        """
        return self.encoder.eval_inference_dist(x, z, param)


    def nll_iw(self, x, nsamples, z, param, recon_type='avg', ns=100):
        """compute the importance weighting estimate of the log-likelihood
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list
            nsamples: Int
                the number of samples required to estimate marginal data likelihood
        Returns: Tensor1
            Tensor1: the estimate of log p(x), shape [batch]
        """
        # compute iw every ns samples to address the memory issue (?)
        # nsamples = 500, ns = 100
        # nsamples = 500, ns = 10
        tmp = []
        for _ in range(1):#range(int(nsamples / ns)):
            # logp_xz + logpz
            log_comp_ll = self.eval_complete_ll(x, z, recon_type)
            # logqz, param is the parameters required to evaluate q(z|x)
            log_infer_ll = self.eval_inference_dist(x, z, param) 
            tmp.append(log_comp_ll - log_infer_ll)
        ll_iw = log_sum_exp(torch.cat(tmp, dim=-1), dim=-1) - math.log(nsamples)
        return ll_iw #-ll_iw
