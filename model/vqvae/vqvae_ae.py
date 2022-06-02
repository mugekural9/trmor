# ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
import torch
from torch import nn
from torch.nn import functional as F
from common.utils import *
from typing import TypeVar, List
Tensor = TypeVar('torch.tensor')

class VQVAE_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init, bidirectional=True):
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

        #self.linear = nn.Linear(args.enc_nh,  args.nz, bias=False)

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
            fwd = last_state[-1].unsqueeze(0)
            bck = last_state[-2].unsqueeze(0)
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)
       
        #z = self.linear(last_state)
        # (batch_size, 1, args.nz)
        #z = z.permute(1,0,2)

        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)

        fwd = fwd.permute(1,0,2)
        bck = bck.permute(1,0,2)

        last_cell = last_cell.permute(1,0,2)
        return last_state, last_cell, None, (fwd,bck)

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

    def forward(self, latents: Tensor,epc, forceid=-1, normalize=True) -> Tensor:

        # latents: (batch_size, 1, enc_nh)
        latents = latents.contiguous()  
        latents_shape = latents.shape
        batch_size, t, emb_dim = latents.shape
        # (batch_size * t, D)
        flat_latents = latents.view(batch_size * t, self.D)  

        # Get the encoding that has the min distance
        # (batch_size * t, 1)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight** 2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  
        #encoding_inds = torch.argmax(F.cosine_similarity(flat_latents.unsqueeze(1), self.embedding.weight, dim=-1),dim=1).unsqueeze(1)
       
        #if forceid> -1:
        #    encoding_inds =  torch.LongTensor([[forceid]])

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        # (batch_size * t, K)
        encoding_one_hot.scatter_(1, encoding_inds, 1) 
        # Quantize the latents
        # (batch_size * t, D)
        quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
        # (batch_size, t, D)
        quantized_latents = quantized_latents.view(latents_shape)  
        # Compute the VQ Losses (avg over all b*t*d)

        '''closs = nn.CosineEmbeddingLoss()
        # Compute the VQ Losses (avg over all b*t*d)
        commitment_loss = closs(quantized_latents.squeeze(1).detach(), latents.squeeze(1), torch.ones(batch_size).to('cuda'))
        embedding_loss = closs(quantized_latents.squeeze(1), latents.squeeze(1).detach(), torch.ones(batch_size).to('cuda'))
        vq_loss = embedding_loss + (self.beta * commitment_loss) '''
        
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents, reduce=False).mean(-1)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach(), reduce=False).mean(-1)
        vq_loss = embedding_loss + (self.beta * commitment_loss)

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        
        # quantized_latents: (batch_size, t, D), vq_loss: scalar
        return quantized_latents.contiguous(), vq_loss, encoding_inds.t()

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
        self.lstm = nn.LSTM(input_size=args.ni + args.incat,
                            hidden_size=args.dec_nh,
                            num_layers=1,
                            batch_first=True)

        # prediction layer
        self.pred_linear = nn.Linear(args.dec_nh+args.outcat, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input, z):

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)
        # (1, batch_size, nz)
        z = z.permute((1,0,2))
        # (1, batch_size, dec_nh)
        c_init = z
        #c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))
        output = self.dropout_out(output)
        # (batch_size, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits

class VQVAE_AE(nn.Module):

    def __init__(self,
                args,
                vocab,
                model_init,
                emb_init,
                dict_assemble_type='concat',
                 **kwargs) -> None:
        super(VQVAE_AE, self).__init__()
        
        self.encoder = VQVAE_Encoder(args, vocab, model_init, emb_init) 
        self.encoder_emb_dim = args.embedding_dim 
        self.num_dicts = args.num_dicts
        self.rootdict_emb_dim = args.rootdict_emb_dim
        #self.rootdict_emb_num = args.rootdict_emb_num
        self.orddict_emb_num = args.orddict_emb_num
        self.dict_assemble_type = dict_assemble_type
        
        #assert (dict_assemble_type=='concat' or (dict_assemble_type =='sum' and self.rootdict_emb_dim == self.encoder_emb_dim)), "If dict assemble type is sum, dict embedding dim should be equal to encoder emb dim"

        self.orddict_emb_dim = int((self.encoder_emb_dim-self.rootdict_emb_dim)/(self.num_dicts-1))
        self.beta = args.beta

        self.decoder = VQVAE_Decoder(args, vocab, model_init, emb_init) 
    
    def recon_loss(self, x, quantized_z, recon_type='avg'):
        # remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        # (batch_size, seq_len, vocab_size)
        #output_logits = self.decoder.forward_no_teacher_forcing(src, quantized_z)
        output_logits = self.decoder(src, quantized_z)
        
        _tgt = tgt.contiguous().view(-1)

        # (batch_size *  seq_len, vocab_size)
        _output_logits = output_logits.reshape(-1, output_logits.size(2))

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


    def loss(self, x: Tensor, epc, **kwargs) -> List[Tensor]:
        # x: (B,T)
        # quantized_inputs: (B, 1, hdim)
        encoder_fhs, fcs, z, (fwd,bck) = self.encoder(x)
        recon_loss, recon_acc = self.recon_loss(x, encoder_fhs, recon_type='sum')
        # (batchsize)
        recon_loss = recon_loss.squeeze(1)
        loss = recon_loss 
        vq_loss = torch.tensor(0); quantized_inds = []
        return loss, recon_loss, vq_loss, recon_acc, quantized_inds, fwd,bck

        
    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = ((pred_tokens == tgt) * (tgt!=0)).sum().item()
        return (acc, pred_tokens)
