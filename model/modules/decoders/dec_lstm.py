# import torch
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from .decoder import DecoderBase
from .decoder_helper import BeamSearchNode

class LSTMDecoder(DecoderBase):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(LSTMDecoder, self).__init__()
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

    def decode(self, input, z):
        """
        Args:
            input: (batch_size, seq_len)
            z: (batch_size, 1, nz)
        """

        # not predicting start symbol
        # sents_len -= 1
        batch_size, _, _ = z.size()
        seq_len = input.size(1)

        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        z_ = z.expand(batch_size, seq_len, self.nz)
        # (batch_size * n_sample, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)

        # (batch_size, nz)
        z = z.view(batch_size, self.nz)

       
        # (1, batch_size, dec_nh)
        #c_init = z.unsqueeze(0)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        output, _ = self.lstm(word_embed, (h_init, c_init))

        output = self.dropout_out(output)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.pred_linear(output)

        return output_logits

    def reconstruct_error(self, x, z):
        """Cross Entropy in the language case
        Args:
            x: (batch_size, seq_len)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        #remove end symbol
        src = x[:, :-1]

        # remove start symbol
        tgt = x[:, 1:]

        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        # (batch_size * n_sample, seq_len, vocab_size)
        output_logits = self.decode(src, z)

        if n_sample == 1:
            tgt = tgt.contiguous().view(-1)
        else:
            # (batch_size * n_sample * seq_len)
            tgt = tgt.unsqueeze(1).expand(batch_size, n_sample, seq_len) \
                     .contiguous().view(-1)

      
        # (batch_size * n_sample * seq_len)
        loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                         tgt)


        # (batch_size, 1)
        return loss.view(batch_size, n_sample, -1).sum(-1)

    def s2s_reconstruct_error(self, y, z):
        #remove end symbol
        src = y[:, :-1]

        # remove start symbol
        tgt = y[:, 1:]

        batch_size, seq_len = src.size()

        # (batch_size, seq_len, vocab_size)
        output_logits = self.decode(src, z)
       
        # (batch_size * seq_len)
        _tgt = tgt.contiguous().view(-1)

        # (batch_size * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * seq_len)
        loss = self.loss(_output_logits,  _tgt)

        # loss: (batch_size): sum the loss over sequence
        return loss.view(batch_size, seq_len).sum(-1), self.accuracy(output_logits, tgt, src)

    def accuracy(self, output_logits, targets, y):
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



    
    
    '''
    def accuracy(self, logits, tgt):
        # logits: (batchsize, T, vocabsize), tgt: (batchsize, T) 
        B, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(logits),2)
        nonpad_tokens = (tgt != self.vocab.word2id['<pad>'])
        correct_tokens = ((pred_tokens == tgt) * nonpad_tokens)
        return correct_tokens.sum().item() , nonpad_tokens.sum().item()
    '''

    def beam_search_decode(self, z, K=5):
        """beam search decoding, code is based on
        https://github.com/pcyin/pytorch_basic_nmt/blob/master/nmt.py

        the current implementation decodes sentence one by one, further batching would improve the speed

        Args:
            z: (batch_size, nz)
            K: the beam width

        Returns: List1
            List1: the decoded word sentence list
        """

        decoded_batch = []
        batch_size, nz = z.size()

        # (1, batch_size, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        # decoding goes sentence by sentence
        for idx in range(batch_size):
            # Start with the start of the sentence token
            decoder_input = torch.tensor([[self.vocab.word2id["<s>"]]], dtype=torch.long, device=self.device)
            decoder_hidden = (h_init[:,idx,:].unsqueeze(1), c_init[:,idx,:].unsqueeze(1))

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0., 1)
            live_hypotheses = [node]

            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < 100:
                t += 1

                # (len(live), 1)
                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=0)

                # (1, len(live), nh)
                decoder_hidden_h = torch.cat([node.h[0] for node in live_hypotheses], dim=1)
                decoder_hidden_c = torch.cat([node.h[1] for node in live_hypotheses], dim=1)

                decoder_hidden = (decoder_hidden_h, decoder_hidden_c)


                # (len(live), 1, ni) --> (len(live), 1, ni+nz)
                word_embed = self.embed(decoder_input)
                word_embed = torch.cat((word_embed, z[idx].view(1, 1, -1).expand(
                    len(live_hypotheses), 1, nz)), dim=-1)

                output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

                # (len(live), 1, vocab_size)
                output_logits = self.pred_linear(output)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses], dtype=torch.float, device=self.device)
                decoder_output = decoder_output + prev_logp.view(len(live_hypotheses), 1, 1)

                # (len(live) * vocab_size)
                decoder_output = decoder_output.view(-1)

                # (K)
                log_prob, indexes = torch.topk(decoder_output, K-len(completed_hypotheses))

                live_ids = indexes // len(self.vocab.word2id)
                word_ids = indexes % len(self.vocab.word2id)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode((decoder_hidden[0][:, live_id, :].unsqueeze(1),
                        decoder_hidden[1][:, live_id, :].unsqueeze(1)),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)

                    if word_id.item() == self.vocab.word2id["</s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)

                live_hypotheses = live_hypotheses_new

                if len(completed_hypotheses) == K:
                    break

            for live in live_hypotheses:
                completed_hypotheses.append(live)

            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.vocab.id2word[n.wordid.item()])
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(self.vocab.id2word[n.wordid.item()])

                utterance = utterance[::-1]

                utterances.append(utterance)

                # only save the top 1
                break

            decoded_batch.append(utterances[0])

        return decoded_batch

    def greedy_decode(self, z, x):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab.word2id["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        end_symbol = torch.tensor([self.vocab.word2id["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        pred_indices = []
        decoder_hiddens = []
        eos_probs = []
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)
            decoder_hiddens.append(decoder_hidden[0])
           
            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            max_index = torch.argmax(output_logits, dim=1)
            probs = F.softmax(output_logits, dim=1)
            probs_indices = torch.argsort(probs, descending=True)
            eos_rank = (probs_indices == 2).nonzero(as_tuple=True)[1].item()
            print(eos_rank)
            eos_probs.append(eos_rank)
            # max_index = torch.multinomial(probs, num_samples=1)
            decoder_input = max_index.unsqueeze(1)
            length_c += 1
            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(max_index[i].item()))
                    pred_indices.append(max_index[i].item())
            mask = torch.mul((max_index != end_symbol), mask)

        if x!= None:
            T = x[:,1:].size(1)
            predT = min(T, len(pred_indices))
            correct_tokens = (torch.tensor(pred_indices)[:predT] == x[:,1:predT+1].cpu()).sum().item()
            _acc = correct_tokens, T, len(pred_indices)
        else:
            _acc = None

        decoder_hiddens = torch.cat(decoder_hiddens)
        print(decoded_batch)
        return decoded_batch, _acc, decoder_hiddens, eos_probs

    def sample_decode(self, z):
        """sampling decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        batch_size = z.size(0)
        decoded_batch = [[] for _ in range(batch_size)]

        # (batch_size, 1, nz)
        c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)

        decoder_hidden = (h_init, c_init)
        decoder_input = torch.tensor([self.vocab["<s>"]] * batch_size, dtype=torch.long, device=self.device).unsqueeze(1)
        end_symbol = torch.tensor([self.vocab["</s>"]] * batch_size, dtype=torch.long, device=self.device)

        mask = torch.ones((batch_size), dtype=torch.uint8, device=self.device)
        length_c = 1
        while mask.sum().item() != 0 and length_c < 100:

            # (batch_size, 1, ni) --> (batch_size, 1, ni+nz)
            word_embed = self.embed(decoder_input)
            word_embed = torch.cat((word_embed, z.unsqueeze(1)), dim=-1)

            output, decoder_hidden = self.lstm(word_embed, decoder_hidden)

            # (batch_size, 1, vocab_size) --> (batch_size, vocab_size)
            decoder_output = self.pred_linear(output)
            output_logits = decoder_output.squeeze(1)

            # (batch_size)
            sample_prob = F.softmax(output_logits, dim=1)
            sample_index = torch.multinomial(sample_prob, num_samples=1).squeeze(1)

            decoder_input = sample_index.unsqueeze(1)
            length_c += 1

            for i in range(batch_size):
                if mask[i].item():
                    decoded_batch[i].append(self.vocab.id2word(sample_index[i].item()))

            mask = torch.mul((sample_index != end_symbol), mask)

        return decoded_batch


