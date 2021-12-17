from itertools import chain
import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .encoder import GaussianEncoderBase
from ..utils import log_sum_exp

class LSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length batching"""
    def __init__(self, args, vocab_size, model_init, emb_init, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)

        # dimension transformation to z (mean and logvar)
        if self.lstm.bidirectional:
            self.linear = nn.Linear(args.enc_nh*2, 2 * args.nz, bias=False)
        else:
            self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)


    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """
        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        _, (last_state, last_cell) = self.lstm(word_embed)
        if self.lstm.bidirectional:
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)

        # (batch_size, T, enc_nh) vq-vae
        # return _, last_state.permute(1,0,2) 
        
        # (batch_size, T, nz)
        # mean, logvar = self.linear(last_state).chunk(2, -1)
        # return mean.permute(1,0,2)
        
        if self.linear is None:
            return None, None, last_state
        else:
            mean, logvar = self.linear(last_state).chunk(2, -1)
        return mean.squeeze(0), logvar.squeeze(0), last_state
       
  
