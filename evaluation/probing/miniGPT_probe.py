# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Emre Can Acikgoz, Muge Kural
# Description: Linear probe
# -----------------------------------------------------------

import torch
import torch.nn as nn
from model.miniGPT.gpt3 import GPT3

class MiniGPT_Probe(nn.Module):
    def __init__(self, args, vocab):
        super(MiniGPT_Probe, self).__init__()
        self.vocab = vocab
        self.token_embedding = args.pretrained_model.token_embedding
        self.position_embedding = args.pretrained_model.position_embedding
        self.embedding_dropout = args.pretrained_model.embedding_dropout
        self.decoder1 = args.pretrained_model.decoder1
        self.decoder2 = args.pretrained_model.decoder2
        self.decoder3 = args.pretrained_model.decoder3
        self.layer_norm = args.pretrained_model.layer_norm
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(args.embed,len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
    
    def forward(self, surf):
        surf = surf[:,:-1]
        b, t = surf.size()
        # forward the GPT model
        # (b,t,embeddim)
        token_embeddings = self.token_embedding(surf)  # each index maps to a learnable vector
        # (1,t,embeddim)
        position_embeddings = self.position_embedding[:, :t, :]  # each position maps to a learnable vector
        # (b,t,embeddim)
        x = self.decoder1(position_embeddings+token_embeddings)
        x = self.decoder2(x)
        x = self.decoder3(x)
        # (b,embeddim, t)
        x = x.permute(0,2,1)
        # (b,embeddim, 1)
        x = self.adaptive_pool(x)
        # (b,1,embeddim)
        x = x.permute(0,2,1)
        # (batchsize, 1, vocab_size)
        output_logits = self.linear(x) 
        return output_logits

    def probe_loss(self, surf, y, plot=False, ratiodict=None,last_iter=False):
        output_logits = self(surf) 
        #(batchsize)
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

class MiniGPT_Probe2(nn.Module):
    def __init__(self, args, vocab):
        super(MiniGPT_Probe2, self).__init__()
        self.vocab = vocab
        self.token_embedding = args.pretrained_model.token_embedding
        self.position_embedding = args.pretrained_model.position_embedding
        self.embedding_dropout = args.pretrained_model.embedding_dropout
        self.decoder1 = args.pretrained_model.decoder1
        self.decoder2 = args.pretrained_model.decoder2
        self.layer_norm = args.pretrained_model.decoder3.layer_norm1
        self.MH_attention3 = args.pretrained_model.decoder3.MH_attention
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(args.embed,len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False)
    
    def forward(self, surf):
        #_, fhs = self.encoder(surf)
        b, t = surf.size()
        #print(f"\n1) src: {surf.shape}, b: {b}, and t: {t} ")

        # forward the GPT model
        token_embeddings = self.token_embedding(surf)  # each index maps to a learnable vector
        #print(f"2) word_embed: {token_embeddings.shape}")
        position_embeddings = self.position_embedding[:, :t, :]  # each position maps to a learnable vector
        #print(f"3) position_embeddings: {position_embeddings.shape}")
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        #print(f"4) embedding_dropout: {x.shape}")
        x = self.decoder1(x)
        #print(f"5) Decoder1 Layer Output: {x.shape}")
        x = self.decoder2(x)
        #print(f"6) Decoder2 Layer Output: {x.shape}")
        x = self.layer_norm(x)
        x = self.MH_attention3(x)
        #print(f"7) MH Attention Layer Output: {x.shape}")

        x = x.permute(0,2,1)
        #print(f"8) Permute 1: {x.shape}")
        x = self.adaptive_pool(x)
        #print(f"9) Adaptive Pool: {x.shape}")
        x = x.permute(0,2,1)
        #print(f"10) Permute 2: {x.shape}")

        # (batchsize, 1, vocab_size)
        output_logits = self.linear(x) 
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