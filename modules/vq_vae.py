# ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *
from modules import LSTMEncoder, LSTMDecoder, OneStepDecoder
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        latents = latents.contiguous()  # [B x T x D]
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

        return quantized_latents.permute(0, 2, 1).contiguous(), vq_loss  # [B x D x 1]

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator
        
    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)

class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final

class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # prev_embed: (B,1,256), encoder_hidden: (B,tx,512), src_mask: (B,1,tx), proj_key: (B,tx, 256), hidden: (1,B,256)

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D] D:256
        
        # context: (B,1,512) attn_probs: (B,1,tx)
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        # (B,1, contextdim+D)
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))            

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        acc = self.accuracy(x,y)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm, acc

    def accuracy(self, logits, tgt):
        # logits: (batchsize, T, vocabsize), tgt: (batchsize, T) 
        B, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(logits),2)
        nonpad_tokens = (tgt != 0)
        correct_tokens = ((pred_tokens == tgt) * nonpad_tokens)
        return correct_tokens.sum().item() , nonpad_tokens.sum().item()

class VQVAE(nn.Module):

    def __init__(self,
                 args,
                 surface_vocab,
                 embedding_dim: int,
                 num_embeddings: int,
                 beta: float = 0.25,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()
        
        '''
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)
     
        self.encoder = LSTMEncoder(args, 
                                    len(surface_vocab), 
                                    model_init, 
                                    emb_init) 
        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        self.decoder = LSTMDecoder(args, 
                                    surface_vocab, 
                                    model_init, 
                                    emb_init) 
        '''
        hidden_size = 256
        num_layers = 1
        dropout = 0.2
        emb_size = 512
        attention = BahdanauAttention(hidden_size)
        self.model = EncoderDecoder(
            Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
            Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
            nn.Embedding(39, emb_size),
            nn.Embedding(39, emb_size),
            Generator(hidden_size, 39))
        self.loss_compute = SimpleLossCompute(self.model.generator, nn.NLLLoss(reduction="sum", ignore_index=0), torch.optim.Adam(self.model.parameters(), lr=0.0003))



    def encode(self, input: Tensor) -> List[Tensor]:
        return self.encoder(input)

    def decode(self, input, z: Tensor) -> Tensor:
        return self.decoder.s2s_reconstruct_error(input, z)

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # input: (B,T)
        B, T = input.shape
        # (B,T)
        src_mask = (input != 0).unsqueeze(1)
        src = input
        trg = input[:, :-1]
        trg_y = input[:, 1:]
        trg_mask = (trg_y != 0)
        ntokens = (trg_y != 0).data.sum().item()
        src_lengths = torch.tensor([T] * B)
        trg_lengths = torch.tensor([T] * B)

        # src: (B,T), tgt: (B,T), lengths: (B), src_mask: (B,1,t), tgt_mask: (B,t)
        out, _, pre_output = self.model(src, trg, src_mask, trg_mask, src_lengths, trg_lengths)
        loss, acc = self.loss_compute(pre_output, trg_y,  src.size(0))
        # (batch_size, t, z)
        # encoding, last_state_vector = self.encode(input)
        # (batch_size, z, t)
        # quantized_inputs, vq_loss = self.vq_layer(encoding)
        # (batch_size, t, z)
        # quantized_inputs = quantized_inputs.permute(0,2,1)

        #recons_loss, recon_acc = self.decode(input, quantized_inputs)
        #recons_loss, recon_acc = self.attndecode(input, last_state_vector, encoding)
        #recons_loss = recons_loss.squeeze(1).mean()
        #loss = recons_loss #+ vq_loss
        return loss, ntokens, acc
        '''
        return {'loss': loss,
                'Reconstruction_Loss': loss,
                'VQ_Loss':0,
                'Reconstruction_Acc': (0,0)}
        '''
  