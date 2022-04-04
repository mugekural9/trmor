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
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)
       
        #z = self.linear(last_state)
        # (batch_size, 1, args.nz)
        #z = z.permute(1,0,2)

        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        last_cell = last_cell.permute(1,0,2)
        return last_state, last_cell, None

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
        
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
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

    def forward_no_teacher_forcing(self, input, z):
        batch_size, _, _ = z.size()
        seq_len = input.size(1)
        sft = nn.Softmax(dim=1)
        # (1, batch_size, dec_nh)
        c_init = z.permute((1,0,2)) 
        h_init = torch.tanh(c_init)
        decoder_hidden = (c_init, h_init)
        i = 0
        bosid = self.vocab.word2id['<s>']
        input = torch.tensor(bosid).repeat(batch_size)
        ol_array = []
        while i < seq_len:
            x = input.unsqueeze(1).to('cuda')
            # (batch_size,1,ni)
            word_embed = self.embed(x)
            word_embed = torch.cat((word_embed, z), -1)
            # output: (batch_size, 1, dec_nh), decoder_hidden: ((1,batch_size,dec_nh), (1,batch_size,dec_nh))
            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)
            # (batch_size, vocab_size)
            output_logits = self.pred_linear(output).squeeze(1)
            ol_array.append(output_logits)
            # batch_size
            input = torch.argmax(sft(output_logits),dim=1) 
            i += 1
        ols = torch.stack(ol_array).permute((1,0,2))
        return ols

class VQVAE(nn.Module):

    def __init__(self,
                args,
                vocab,
                model_init,
                emb_init,
                dict_assemble_type='concat',
                 **kwargs) -> None:
        super(VQVAE, self).__init__()
        
        self.encoder = VQVAE_Encoder(args, vocab, model_init, emb_init) 
        self.encoder_emb_dim = args.embedding_dim 
        self.num_dicts = args.num_dicts
        self.rootdict_emb_dim = args.rootdict_emb_dim
        self.rootdict_emb_num = args.rootdict_emb_num
        self.orddict_emb_num = args.orddict_emb_num
        self.dict_assemble_type = dict_assemble_type
        
        assert (dict_assemble_type=='concat' or (dict_assemble_type =='sum' and self.rootdict_emb_dim == self.encoder_emb_dim)), "If dict assemble type is sum, dict embedding dim should be equal to encoder emb dim"

        if dict_assemble_type =='concat':
            self.orddict_emb_dim = int((self.encoder_emb_dim-self.rootdict_emb_dim)/(self.num_dicts-1))
        else:
            self.orddict_emb_dim = self.rootdict_emb_dim
        
        self.beta = args.beta

        #self.linear_root = nn.Linear(self.encoder_emb_dim,  self.rootdict_emb_dim, bias=True)
        self.vq_layer_root = VectorQuantizer(self.rootdict_emb_num,
                                        self.rootdict_emb_dim,
                                        self.beta)

        #other
        self.ord_linears = nn.ModuleList([])
        self.ord_vq_layers = nn.ModuleList([])
      
        for i in range(self.num_dicts-1):
            self.ord_linears.append(nn.Linear(self.encoder_emb_dim,  self.orddict_emb_dim, bias=True))
            self.ord_vq_layers.append(VectorQuantizer(self.orddict_emb_num,
                                        self.orddict_emb_dim,
                                        self.beta))

        self.decoder = VQVAE_Decoder(args, vocab, model_init, emb_init) 
    
    def vq_loss(self,x, epc):
        # fhs: (B,1,hdim)
        fhs, fcs, z = self.encoder(x)
        #_fhs = self.linear_root(fhs)
        vq_vectors = []; vq_losses = []; vq_inds = []
        # quantized_input: (B, 1, rootdict_emb_dim)
        quantized_input, vq_loss, quantized_inds = self.vq_layer_root(fhs[:,:,:320],epc)
        vq_vectors.append(quantized_input)
        vq_losses.append(vq_loss)
        vq_inds.append(quantized_inds)
        # quantize thru ord dicts
        for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
            _fhs =  linear(fhs)
            # quantized_input: (B, 1, orddict_emb_dim)
            quantized_input, vq_loss, quantized_inds = vq_layer(_fhs,epc)
            vq_vectors.append(quantized_input)
            vq_losses.append(vq_loss)
            vq_inds.append(quantized_inds)
        
        if self.dict_assemble_type == 'sum':
            for i in range(1, len(vq_vectors)):
                vq_vectors[0] += vq_vectors[i]
            vq_vectors = vq_vectors[0]
        else:
            # concat quantized vectors
            vq_vectors = torch.cat(vq_vectors,dim=2)
        
        vq_loss =  torch.stack(vq_losses).mean()

        return vq_vectors, vq_loss, vq_inds,  fhs

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
        quantized_inputs, vq_loss, quantized_inds, encoder_fhs = self.vq_loss(x, epc)
        recon_loss, recon_acc = self.recon_loss(x, quantized_inputs)
        # avg over batches
        recon_loss = recon_loss.mean()
        loss = recon_loss + vq_loss
        return loss, recon_loss, vq_loss, recon_acc, quantized_inds, encoder_fhs

        
    def accuracy(self, output_logits, tgt):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = ((pred_tokens == tgt) * (tgt!=0)).sum().item()
        return (acc, pred_tokens)