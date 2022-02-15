# ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
import torch
from torch import nn
from torch.nn import functional as F
from common.utils import *
from typing import TypeVar, List
Tensor = TypeVar('torch.tensor')

class VQVAE_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=False):
        super(VQVAE_Encoder, self).__init__()
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
        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        return last_state

class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        # latents: (B x T x D)
        latents = latents.contiguous()  
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BT x D]

        # (BT x K)
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  

        # (BT x 1)
        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  
        #print(encoding_inds.t())
        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        # (BT x K)
        encoding_one_hot.scatter_(1, encoding_inds, 1) 

        # Quantize the latents
        # (BT, D)
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  
        # (B x 1 x D)
        quantized_latents = quantized_latents.view(latents_shape)  

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.contiguous(), vq_loss  # [B x D x 1]


class VQVAE_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(VQVAE_Decoder, self).__init__()
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
        #self.trans_linear = nn.Linear(args.nz, args.dec_nh, bias=False)

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

        z_ = z.expand(batch_size, seq_len, self.nz)
     
        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)
        # (batch_size * n_sample, nz)
        z = z.view(batch_size * n_sample, self.nz)

        # (1, batch_size, dec_nh)
        c_init = z.unsqueeze(0) #self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))
        output = self.dropout_out(output)
        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits

class VQVAE(nn.Module):

    def __init__(self,
                args,
                vocab,
                model_init,
                emb_init,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()
        
        self.encoder = VQVAE_Encoder(args, vocab, model_init, emb_init) 

        self.vq_layer = VectorQuantizer(args.num_embeddings,
                                        args.embedding_dim,
                                        args.beta)

        self.decoder = VQVAE_Decoder(args, vocab, model_init, emb_init) 

        self.beta = args.beta

        #self.multi_vq_layer = MultiVectorQuantizer(num_embeddings,
        #                                args.nz,
        #                                self.beta)

    def vq_loss(self, x):
        # fhs: (B,1,hdim)
        fhs = self.encoder(x)
        # quantized_inputs: (B, 1, hdim)
        quantized_inputs, vq_loss = self.vq_layer(fhs)
        return quantized_inputs, vq_loss

    def recon_loss(self, x, quantized_z, recon_type='avg'):
        # remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decoder(src, quantized_z)
        _tgt = tgt.contiguous().view(-1)

        # (batch_size * 1 * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * 1 * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, 1, seq_len)
        recon_loss = recon_loss.view(batch_size, 1, -1)

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


    def loss(self, x: Tensor, **kwargs) -> List[Tensor]:
        # x: (B,T)
        # quantized_inputs: (B, 1, hdim)
        quantized_inputs, vq_loss = self.vq_loss(x)
        recon_loss, recon_acc = self.recon_loss(x, quantized_inputs)
        # avg over batches
        recon_loss = recon_loss.mean()
        loss = recon_loss + vq_loss
        return loss, recon_loss, vq_loss, recon_acc

        
    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        return acc