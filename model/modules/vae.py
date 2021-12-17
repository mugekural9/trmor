import math
import torch
import torch.nn as nn
import numpy as np
from .utils import log_sum_exp

general_last_states = []

class VAE(nn.Module):
    def __init__(self, encoder, decoder, args):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.args = args

        self.nz = args.nz

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)

        self.prior = torch.distributions.normal.Normal(loc, scale)

    def encode(self, x,  nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        return self.encoder.encode(x, nsamples)

    def decode(self, z, strategy, surf=None, K=5):
        """generate samples from z given strategy

        Args:
            z: [batch, nsamples, nz]
            strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter

        Returns: List1
            List1: a list of decoded word sequence
        """

        if strategy == "beam":
            return self.decoder.beam_search_decode(z, K)
        elif strategy == "greedy":
            return self.decoder.greedy_decode(z, surf)
        elif strategy == "sample":
            return self.decoder.sample_decode(z)
        else:
            raise ValueError("the decoding strategy is not supported")

    def reconstruct(self, x, decoding_strategy="greedy", surf=None, K=5):
        """reconstruct from input x

        Args:
            x: (batch, *)
            decoding_strategy: "beam" or "greedy" or "sample"
            K: the beam width parameter (if applicable)

        Returns: List1
            List1: a list of decoded word sequence
        """
        z = self.sample_from_inference(x).squeeze(1)

        #return z, self.decode(z, decoding_strategy, K)
        return self.decode(z, decoding_strategy, surf, K)
    
    def s2s_loss(self, x, y, nsamples=1):
        """
        Args:
            x: if the data is constant-length, x is the data tensor with
                shape (batch, *). Otherwise x is a tuple that contains
                the data tensor and length list

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: total loss [batch]
            Tensor2: reconstruction loss shape [batch]
            Tensor3: KL loss shape [batch]
        """
        # z: (batchsize, 1, nz)
        z, KL, _ = self.encode(x, 1)

        # reconstruct_err: (batch_size)
        reconstruct_err, acc = self.decoder.s2s_reconstruct_error(y, z)
        return reconstruct_err, acc 

    def linear_probe_loss(self, x, y, task, nsamples=1):
        # z: (batchsize, 1, nz), last_states: (1, batchsize, enc_nh)
        z, _, last_states = self.encode(x, nsamples)
        # general_last_states.append(last_states.squeeze(0))
        # (1, batchsize, vocab_size)
        output_logits = self.probe_linear(last_states) 
        # (batch_size, 1, vocab_size)
        output_logits = output_logits.permute(1,0,2)
        loss = self.closs(output_logits.squeeze(1), y.squeeze(1))
        acc = self.accuracy(output_logits, y, x, task)
        return loss, acc, general_last_states

    def accuracy(self, output_logits, targets, x, task):
        surface_vocab, feature_vocab, pos_vocab, polar_vocab, tense_vocab, surfpos_vocab = self.vocab

        B, T = targets.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        correct_tokens = (pred_tokens == targets)
        wrong_tokens = (pred_tokens != targets)
        wrong_predictions = []
        correct_predictions = []
        for i in range(len(x)):
            surf_str =  ''.join(surface_vocab.decode_sentence(x[i])[1:-1])

            if task == 'surf2polar':
                target_vocab = polar_vocab
            elif task == 'surf2surfpos' or task == 'root2pos':
                target_vocab = surfpos_vocab
            elif task == 'surf2tense':
                target_vocab = tense_vocab
            target  = target_vocab.decode_sentence(targets[i])[0]
            pred = target_vocab.id2word(pred_tokens[i].item())
            if target != pred:
                wrong_predictions.append('%s target: %s pred: %s' % (surf_str, target, pred))
            else:
                correct_predictions.append('%s target: %s pred: %s' % (surf_str, target, pred))
        acc = correct_tokens.sum().item(), B, wrong_tokens.sum().item(), wrong_predictions, correct_predictions
        return  acc

    def sample_from_inference(self, x, nsamples=1):
        """perform sampling from inference net
        Returns: Tensor
            Tensor: samples from infernece nets with
                shape (batch_size, nsamples, nz)
        """
        z, _ = self.encoder.sample(x, 1)

        return z