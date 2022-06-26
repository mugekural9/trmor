# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Character-based Variational Autoencoder 
# -----------------------------------------------------------

import math
from pprint import pprint
import torch
import torch.nn as nn
import numpy as np
from common.utils import log_sum_exp
from torch.nn import functional as F

class MSVED_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(MSVED_Encoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz

        self.embed = nn.Embedding(len(vocab.word2id), args.ni)

        self.gru = nn.GRU(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)

        # dimension transformation to z
        if self.gru.bidirectional:
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

        _, last_state = self.gru(word_embed)
        if self.gru.bidirectional:
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

class MSVED_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(MSVED_Decoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        self.char_embed = nn.Embedding(len(vocab.word2id), 300, padding_idx=0)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)

        # for initializing hidden state and cell
        self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

        # concatenate z with input
        self.gru = nn.GRU(input_size=650, # self.char_embed+ self.ni
                            hidden_size=256,
                            num_layers=1,
                            batch_first=True)

        self.attn = nn.Linear(300+ 150+ 256, 11)
        self.attn_combine = nn.Linear(650, 650)

        # prediction layer
        self.pred_linear = nn.Linear(args.dec_nh, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.char_embed.weight)

    def forward(self, input, z, hidden, tag_embeddings):

        batch_size, n_sample, _ = z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        embedded = self.char_embed(input)
        embedded = self.dropout_in(embedded)
        z_ = z.expand(batch_size, seq_len, self.nz)
        embedded = torch.cat((embedded, z_), -1)

        # (batchsize,1, tagsize)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        # (batchsize,1, 200)
        attn_applied = torch.bmm(attn_weights,
                                 tag_embeddings)
        # (batchsize,1, z+ni+tag_context_size)
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)
        output = F.relu(output)
        output, hidden = self.gru(output, torch.permute(hidden, (1,0,2)))
        # (batch_size, 1, vocab_size)
        output_logits = self.pred_linear(output)
        hidden = torch.permute(hidden, (1,0,2))
        return output_logits, hidden, attn_weights

class MSVED(nn.Module):
    def __init__(self, args, surf_vocab, tag_vocabs, model_init, emb_init):
        super(MSVED, self).__init__()
        self.encoder = MSVED_Encoder(args, surf_vocab, model_init, emb_init)
        self.decoder = MSVED_Decoder(args, surf_vocab, model_init, emb_init)

        self.args = args
        self.nz = args.nz

        self.tag_embed_dim = 200
        self.dec_nh = 256
        self.char_emb_dim = 300
        self.z_to_dec = nn.Linear(self.nz, 256)
        self.tag_to_dec = nn.Linear(self.tag_embed_dim, 256)

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        self.tag_embeddings = nn.ModuleList([])
        self.classifiers = nn.ModuleList([])


        # Discriminative classifiers for q(y|x)
        for key,keydict in tag_vocabs.items():
            self.classifiers.append(nn.Linear(256*2, len(keydict)))


        for key,keydict in tag_vocabs.items():
            print(key, len(keydict))
            self.tag_embeddings.append(nn.Embedding(len(keydict), self.tag_embed_dim))

    def classifier_loss(self, enc_nh, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss):
        sft = nn.Softmax(dim=2)
        loss = nn.CrossEntropyLoss()

        # (enc_nh: batchsize,1, 256*2)
        tags = [case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss]
        probs =[]
        xloss = torch.tensor(0.0).to('cuda')
        gumbel_tag_embeddings = []
        for i in range(len(self.classifiers)):
            # (batchsize,1,tagvocabsize)
            logits = self.classifiers[i](enc_nh)
            xloss += loss(logits.squeeze(1), tags[i].squeeze(1))
            probs.append(sft(logits))
            # (batchsize,tagvocabsize)
            gumbel_logits = F.gumbel_softmax(logits, tau=1, hard=False).squeeze(1)
            gumbel_tag_embeddings.append(torch.matmul(gumbel_logits, self.tag_embeddings[i].weight).unsqueeze(1))

        # (batchsize,11,tag_embed_dim)
        gumbel_tag_embeddings = torch.cat(gumbel_tag_embeddings, dim=1)

    def loss(self, x, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, reinflect_surf, kl_weight, mode='train'):
     
        mu, logvar, encoder_fhs = self.encoder(x)
       
        classifier_loss = self.classifier_loss(encoder_fhs, case,polar,mood,evid,pos,per,num,tense,aspect,inter,pos)

        # (batchsize, 1, nz)
        z = self.reparameterize(mu, logvar)
       


        #(batchsize,1,tag_embed_dim)
        case_embed  = ((case!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[0](case))
        polar_embed = ((polar!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[1](polar))
        mood_embed = ((mood!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[2](mood))
        evid_embed = ((evid!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[3](evid))
        pos_embed = ((pos!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[4](pos))
        per_embed = ((per!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[5](per))
        num_embed = ((num!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[6](num))
        tense_embed = ((tense!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[7](tense))
        aspect_embed = ((aspect!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[8](aspect))
        inter_embed = ((inter!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[9](inter))
        poss_embed = ((poss!=0).unsqueeze(1).repeat(1,1,self.tag_embed_dim) * self.tag_embeddings[10](poss))

        tag_all_embed = case_embed + polar_embed + mood_embed + evid_embed + pos_embed + per_embed + num_embed + tense_embed + aspect_embed + inter_embed + poss_embed
        #TODO: add bias and pass tru activation

        tag_attention_values = torch.cat((case_embed, polar_embed, mood_embed, evid_embed, pos_embed, per_embed, num_embed, tense_embed, aspect_embed, inter_embed, poss_embed),dim=1)

        dec_h0 = self.tag_to_dec(tag_all_embed) + self.z_to_dec(z)
        #TODO: add bias and pass tru activation

        if mode == 'train':
            recon_loss, recon_acc = self.recon_loss(reinflect_surf, z, dec_h0, tag_attention_values, recon_type='sum')
        else:
            recon_loss, recon_acc = self.recon_loss_test(reinflect_surf, z, dec_h0, tag_attention_values, recon_type='sum')


        # (batchsize)
        kl_loss = self.kl_loss(mu,logvar)

        # (batchsize)
        recon_loss = recon_loss.squeeze(1)#.mean()
        #kl_loss = kl_loss.mean()
        loss = recon_loss + kl_weight * kl_loss
        return loss, recon_loss, kl_loss, recon_acc, encoder_fhs

    def kl_loss(self, mu, logvar):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return KL

    def recon_loss(self, y, z, decoder_hidden, tag_attention_values, recon_type='avg'):
        #remove end symbol
        src = y[:, :-1]
        # remove start symbol
        tgt = y[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        decoder_input = tgt[:,0].unsqueeze(1)
        output_logits = []
        for di in range(seq_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_attention_values)
            output_logits.append(decoder_output)
            decoder_input = tgt[:,di].unsqueeze(1)  # Teacher forcing
        
        # (batchsize, seq_len, vocabsize)
        output_logits = torch.cat(output_logits,dim=1)

        _tgt = tgt.contiguous().view(-1)
        
        # (batch_size  * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, seq_len)
        recon_loss = recon_loss.view(batch_size, n_sample, -1)

        # (batch_size, 1)
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


    def recon_loss_test(self, y, z, decoder_hidden, tag_attention_values, recon_type='avg'):
        #remove end symbol
        src = y[:, :-1]
        # remove start symbol
        tgt = y[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        decoder_input = tgt[:,0].unsqueeze(1)
        output_logits = []
        for di in range(seq_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_attention_values)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
        # (batchsize, seq_len, vocabsize)
        output_logits = torch.cat(output_logits,dim=1)

        _tgt = tgt.contiguous().view(-1)
        
        # (batch_size  * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, seq_len)
        recon_loss = recon_loss.view(batch_size, n_sample, -1)

        # (batch_size, 1)
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
        #print(''.join(self.decoder.vocab.decode_sentence(pred_tokens[0])))
        return acc



