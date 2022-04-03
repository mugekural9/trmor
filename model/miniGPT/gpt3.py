"""
##### Output Example #####
Lenght of batch: torch.Size([32, 6])
len vocab.word2id: 39
1) x: torch.Size([32, 6])
2) src: torch.Size([32, 5]) and tgt: torch.Size([32, 5])
3) word_embed: torch.Size([32, 5, 128])
4) position_embeddings: torch.Size([1, 5, 128])
5) embedding_dropout: torch.Size([32, 5, 128])
6) Decoder Input: torch.Size([32, 5, 128])
7) Decoder Layer Norm1: torch.Size([32, 5, 128])
8) Multi-Head Attention Input: torch.Size([32, 5, 128])
9) K: torch.Size([32, 16, 5, 8]), Q: torch.Size([32, 16, 5, 8]), V: torch.Size([32, 16, 5, 8])
10) Attention: torch.Size([32, 16, 5, 5])
tensor([[ 0.6997,    -inf,    -inf,    -inf,    -inf],
        [-1.5819, -0.3455,    -inf,    -inf,    -inf],
        [ 0.2606, -0.6503,  0.6313,    -inf,    -inf],
        [-1.2693, -0.6447,  0.4583, -0.2119,    -inf],
        [ 0.6234, -0.1545,  0.2546, -0.0763, -0.2299]],
       grad_fn=<SelectBackward0>)
11) Attention: torch.Size([32, 16, 5, 5])
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2251, 0.7749, 0.0000, 0.0000, 0.0000],
        [0.3508, 0.1411, 0.5082, 0.0000, 0.0000],
        [0.0879, 0.1642, 0.4948, 0.2531, 0.0000],
        [0.3253, 0.1494, 0.2250, 0.1616, 0.1386]], grad_fn=<SelectBackward0>)
12) Softmax Attention: torch.Size([32, 16, 5, 5])
13) Attention: torch.Size([32, 16, 5, 5])
14) Attention Output: torch.Size([32, 16, 5, 8])
15) Multi-Head Attention Output: torch.Size([32, 5, 128])
16) Full Connected: torch.Size([32, 5, 128])
17) Dropout: torch.Size([32, 5, 128])
18) Multi-Head Layer Output: torch.Size([32, 5, 128])
19) Decoder Layer Norm2: torch.Size([32, 5, 128])
20) Decoder Layer Output: torch.Size([32, 5, 128])
21) Decoder Layer Output: torch.Size([32, 5, 128])
22) layer_norm: torch.Size([32, 5, 128])
23) logits: torch.Size([32, 5, 39])
24) _tgt: torch.Size([160])
25) _output_logits: torch.Size([160, 39])
26) loss: torch.Size([])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multihead_attention import MultiHead_Masked_SelfAttention


class Decoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, block_size=128, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4):
        super(Decoder, self).__init__()
        self.MH_attention = MultiHead_Masked_SelfAttention(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*expand_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*expand_ratio, embed_dim),
            nn.Dropout(p=residual_dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        res1 = x
        #print(f"6) Decoder Input: {x.shape}")
        x = self.layer_norm1(x)
        #print(f"7) Decoder Layer Norm1: {x.shape}")
        x = self.MH_attention(x)
        #print(f"18) Multi-Head Layer Output: {x.shape}")
        x = x + res1

        res2 = x
        x = self.layer_norm2(x)
        #print(f"19) Decoder Layer Norm2: {x.shape}")
        x = self.feed_forward(x)
        #print(f"20) Decoder Layer Output: {x.shape}")
        x = x + res2

        return x


class GPT3(nn.Module):
    def __init__(self, vocab, num_layers, embed_dim, num_heads=8, block_size=128, embedding_dropout_rate=0.1, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4):
        super(GPT3, self).__init__()
        self.vocab = vocab
        self.token_embedding = nn.Embedding(num_embeddings=len(vocab.word2id), embedding_dim=embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_rate)
        self.decoder1 = Decoder(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate, expand_ratio=expand_ratio)
        self.decoder2 = Decoder(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate, expand_ratio=expand_ratio)
        self.decoder3 = Decoder(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate, expand_ratio=expand_ratio)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, len(vocab.word2id), bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):
        #print(f"len vocab.word2id: {len(self.vocab.word2id)}")
        # pre-process
        #print(f"1) x: {x.shape}")
        src = x[:, :-1]  # remove end symbol
        tgt = x[:, 1:]  # remove start symbol
        b, t = src.size()
        #print(f"2) src: {src.shape} and tgt: {tgt.shape}")

        # forward the GPT model
        token_embeddings = self.token_embedding(src)  # each index maps to a learnable vector
        #print(f"3) word_embed: {token_embeddings.shape}")
        position_embeddings = self.position_embedding[:, :t, :]  # each position maps to a learnable vector
        #print(f"4) position_embeddings: {position_embeddings.shape}")
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        #print(f"5) embedding_dropout: {x.shape}")
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        #print(f"21) Decoder Layer Output: {x.shape}")
        x = self.layer_norm(x)
        #print(f"22) layer_norm: {x.shape}")
        logits = self.head(x)
        #print(f"23) logits: {logits.shape}")

        _tgt = tgt.contiguous().view(-1)
        #print(f"24) _tgt: {_tgt.shape}")
        _output_logits = logits.view(-1, logits.size(-1))
        #print(f"25) _output_logits: {_output_logits.shape}")
        # calculate the loss
        loss = F.cross_entropy(_output_logits, _tgt)
        #print(f"26) loss: {loss.shape}")

        return loss, self.accuracy(logits, tgt), logits

    def log_probability(self, x):
        return -self.forward(x)[0]

    def accuracy(self, output_logits, targets):
        # output_logits: (B, T, vocab_size), targets: (B,T)
        surface_vocab = self.vocab
        B, T = targets.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        correct_tokens = (pred_tokens == targets)
        wrong_tokens = (pred_tokens != targets)
        wrong_predictions = []
        correct_predictions = []
        for i in range(B):
            target  = ''.join(surface_vocab.decode_sentence(targets[i]))
            pred = ''.join(surface_vocab.decode_sentence(pred_tokens[i]))
            if target != pred:
                wrong_predictions.append('target: %s pred: %s' % (target, pred))
            else:
                correct_predictions.append('target: %s pred: %s' % (target, pred))
        acc = correct_tokens.sum().item(), B*T, wrong_tokens.sum().item(), wrong_predictions, correct_predictions
        return  acc