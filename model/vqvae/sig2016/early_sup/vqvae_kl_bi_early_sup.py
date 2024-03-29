# ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
from bdb import Breakpoint
import torch, json
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

        _, (last_state, last_cell) = self.lstm(word_embed)
        if self.lstm.bidirectional:
            fwd = last_state[-1].unsqueeze(0)
            bck = last_state[-2].unsqueeze(0)
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)
            
        #mean, logvar = self.linear(last_state).chunk(2, -1)
        
        mean, logvar = self.linear(fwd).chunk(2, -1)
        fwd = fwd.permute(1,0,2)
        bck = bck.permute(1,0,2)


        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        last_cell = last_cell.permute(1,0,2)

        return last_state, last_cell, None , mean.squeeze(0), logvar.squeeze(0), fwd,bck

## Model
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

    def forward(self, gold_encoding_inds, latents: Tensor, epc, forceid=-1, normalize=True, use_golds=False) -> Tensor:

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
        
        
        if use_golds is False or gold_encoding_inds is None: # infer tags
            encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
            encoding_one_hot.scatter_(1, encoding_inds, 1) 
            quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
            quantized_latents = quantized_latents.view(latents_shape)  
            commitment_loss = F.mse_loss(quantized_latents.detach(), latents, reduce=False).mean(-1)
            embedding_loss = F.mse_loss(quantized_latents, latents.detach(), reduce=False).mean(-1)
            vq_loss = embedding_loss + (self.beta * commitment_loss)
            quantized_latents = latents + (quantized_latents - latents).detach()

            if gold_encoding_inds is not None:
                correct_entry_selection = torch.sum(gold_encoding_inds == encoding_inds).item()
                total_entry_selection = gold_encoding_inds.size(0)
            else:
                correct_entry_selection = 0
                total_entry_selection = 0
            return quantized_latents.contiguous(), vq_loss, encoding_inds.t(), (correct_entry_selection, total_entry_selection)
        else:
            encoding_one_hot = torch.zeros(gold_encoding_inds.size(0), self.K, device=latents.device)
            encoding_one_hot.scatter_(1, gold_encoding_inds, 1) 
            gold_quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
            gold_quantized_latents = gold_quantized_latents.view(latents_shape)  
            commitment_loss = F.mse_loss(gold_quantized_latents.detach(), latents, reduce=False).mean(-1)
            embedding_loss = F.mse_loss(gold_quantized_latents, latents.detach(), reduce=False).mean(-1)
            vq_loss = embedding_loss + (self.beta * commitment_loss)
            

            #encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
            #encoding_one_hot.scatter_(1, encoding_inds, 1) 
            #quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
            #quantized_latents = quantized_latents.view(latents_shape)  

            #encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
            #encoding_one_hot.scatter_(1, encoding_inds, 1) 
            #quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
            #quantized_latents = quantized_latents.view(latents_shape)  
            correct_entry_selection = torch.sum(gold_encoding_inds == encoding_inds).item()
            total_entry_selection = gold_encoding_inds.size(0)
            return gold_quantized_latents.contiguous(), vq_loss, gold_encoding_inds.t(), (correct_entry_selection, total_entry_selection)
            
            #else: #test   
            #    gold_quantized_latents = latents + (gold_quantized_latents - latents).detach()
            #    return gold_quantized_latents.contiguous(), vq_loss, gold_encoding_inds.t()
            
            

class VQVAE_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, vocab, model_init, emb_init):
        super(VQVAE_Decoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.incat = args.incat
        self.vocab = vocab
        self.device = args.device

        # no padding when setting padding_idx to -1
        self.embed = nn.Embedding(len(vocab.word2id), args.ni, padding_idx=0)
        self.dropout_in = nn.Dropout(args.dec_dropout_in)
        self.dropout_out = nn.Dropout(args.dec_dropout_out)

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

    def forward(self, input, z, hidden):
        
        # root_z: (batchsize,1,128) , suffix_z: (batchsize,1, self.incat)
        root_z, suffix_z = z
        # (1, batch_size, nz)
        batch_size, _, _ = root_z.size()
        seq_len = input.size(1)
        
        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        z_ = suffix_z.expand(batch_size, seq_len, self.incat)
        # (batch_size, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)
       
       
        output, hidden = self.lstm(word_embed, hidden)
        # (batch_size, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits, hidden

    def forward_yedek(self, input, z, hidden):
        ##v4
        root_z, suffix_z = z
        batch_size, _, _ = root_z.size()
        seq_len = input.size(1)
        
        # (batch_size, seq_len, ni)
        #word_embed = self.embed(input)
        #word_embed = self.dropout_in(word_embed)
        
        #z_ = suffix_z.expand(batch_size, seq_len, self.incat)# 64
        # (batch_size, seq_len, ni + nz)
        #word_embed = torch.cat((word_embed, z_), -1)
        
        # (1, batch_size, nz)
        #root_z = root_z.permute((1,0,2))
        #z = z.permute((1,0,2))
        
        # (1, batch_size, dec_nh)
        #c_init = root_z
        #h_init = torch.tanh(c_init)

        sft = nn.Softmax(dim=2)

        i = 0
        #decoder_hidden = (h_init, c_init)
        copied = []
        bosid = self.vocab.word2id['<s>']
        input = torch.tensor(bosid)
        input = input.repeat(batch_size,1).to('cuda')
        logitss = []
        while True:
            # (1,1,ni)
            word_embed = self.embed(input)
            word_embed = torch.cat((word_embed, suffix_z), -1)
            # output: (1,1,dec_nh)
            output, hidden = self.lstm(word_embed, hidden)
            # (1, vocab_size)
            output_logits = self.pred_linear(output)
            input = torch.argmax(sft(output_logits),dim=2) 
            logitss.append(output_logits)
            #char = self.vocab.id2word(input.item())
            #copied.append(char)
            i+=1
            if i == seq_len:
               break
        output_logits =  torch.cat(logitss,dim=1)
        return output_logits, hidden

class VQVAE(nn.Module):

    def __init__(self,
                args,
                surf_vocab,
                tag_vocabs,
                model_init,
                emb_init,
                dict_assemble_type='concat',
                 **kwargs) -> None:
        super(VQVAE, self).__init__()
        
        self.encoder = VQVAE_Encoder(args, surf_vocab, model_init, emb_init, bidirectional=True) 
        self.num_dicts = args.num_dicts
        self.dict_assemble_type = dict_assemble_type
        self.nz = args.nz
        self.z_to_dec = nn.Linear(self.nz, args.dec_nh)
        self.orddict_emb_dim   = int(args.enc_nh/self.num_dicts)
            
        self.beta = args.beta
        self.ord_vq_layers = nn.ModuleList([])

        for key,values in tag_vocabs.items():
            self.ord_vq_layers.append(VectorQuantizer(len(values),
                                        self.orddict_emb_dim,
                                        self.beta))
        self.decoder = VQVAE_Decoder(args, surf_vocab, model_init, emb_init) 
    

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

    def kl_loss(self, mu, logvar):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return KL


    def vq_loss(self,x, tags, use_golds, epc, mode):
        # fhs: (B,1,hdim)
        
        # kl version
        fhs, _, _, mu, logvar, fwd,bck = self.encoder(x)

        if mode =='train':
            #(batchsize,1,128)
            _root_fhs = self.reparameterize(mu, logvar)
        else:
            _root_fhs = torch.permute(mu.unsqueeze(0), (1,0,2)).contiguous()
        
        kl_loss = self.kl_loss(mu,logvar)
        
        vq_vectors = []; vq_losses = []; vq_inds = []
        _root_fhs = self.z_to_dec(_root_fhs)

        vq_vectors.append(_root_fhs)
       
        # quantize thru ord dicts
        i=0
        #for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
        for vq_layer in self.ord_vq_layers:
            #_fhs =  linear(fwd_fhs)
            # quantized_input: (B, 1, orddict_emb_dim)
            if tags is not None:
                quantized_input, vq_loss, quantized_inds, selections = vq_layer(tags[i], bck[:,:,i*self.orddict_emb_dim:(i+1)*self.orddict_emb_dim],epc, use_golds=use_golds)
            else:
                quantized_input, vq_loss, quantized_inds, selections = vq_layer(None, bck[:,:,i*self.orddict_emb_dim:(i+1)*self.orddict_emb_dim],epc, use_golds=use_golds)

            vq_vectors.append(quantized_input)
            vq_losses.append(vq_loss)
            vq_inds.append(quantized_inds)
            i+=1
      
        #logdetmaxloss =  self.detmax.loss(0.2,0.2,torch.cat(vq_vectors[1:],dim=2))
        #print(logdetmaxloss)
        vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2)) 
        # (batchsize, numdicts)
        vq_loss =  torch.cat(vq_losses,dim=1)
        # (batchsize)
        vq_loss = vq_loss.sum(-1)
        # (batchsize,1)
        vq_loss = vq_loss.unsqueeze(1)

        suffix_codes = []
        for i in range(vq_loss.shape[0]):
            suffix_code = "" 
            for j in range(0, len(vq_inds)):
                suffix_code += '-' + str((vq_inds[j][0][i]).item()) 
            suffix_codes.append(suffix_code)
        
        return vq_vectors, vq_loss, vq_inds,  fhs, [], suffix_codes, kl_loss, torch.tensor(0.0), selections


    def recon_loss(self, x, dec_h0, quantized_z, dict_codes=None, recon_type='avg'):
        # remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()

        output_logits, _ = self.decoder(src, quantized_z, dec_h0)
        # (batch_size *  seq_len, vocab_size)
        _output_logits = output_logits.reshape(-1, output_logits.size(2))
        _tgt = tgt.contiguous().view(-1)

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
        recon_acc, recon_preds  = self.accuracy(output_logits, tgt, dict_codes)
        return recon_loss, recon_acc, recon_preds

    def recon_loss_test(self, x, decoder_hidden, quantized_z, dict_codes=None, recon_type='avg'):
        # remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        
        decoder_input = src[:,0].unsqueeze(1)
        output_logits = []
        for di in range(seq_len):
            decoder_output, decoder_hidden  = self.decoder(
                decoder_input, quantized_z, decoder_hidden)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
        # (batchsize, seq_len, vocabsize)
        output_logits = torch.cat(output_logits,dim=1)
        # (batch_size *  seq_len, vocab_size)
        _output_logits = output_logits.reshape(-1, output_logits.size(2))
        _tgt = tgt.contiguous().view(-1)

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
        recon_acc, recon_preds  = self.accuracy(output_logits, tgt, dict_codes)
        return recon_loss, recon_acc, recon_preds


    def loss(self, x: Tensor, tags, kl_weight, epc, mode='train', **kwargs) -> List[Tensor]:
        # x: (B,T)
        # quantized_inputs: (B, 1, hdim)
        if mode == 'train':
            use_golds = True
        else:
            use_golds = False
        quantized_inputs, vq_loss, quantized_inds, encoder_fhs, dict_codes, suffix_codes, kl_loss, logdet, selections = self.vq_loss(x, tags, use_golds, epc, mode)
      
        root_z, suffix_z = quantized_inputs
        root_z = root_z.permute((1,0,2))
        # (1, batch_size, dec_nh)
        c_init = root_z
        h_init = torch.tanh(c_init)
        dec_h0 = (h_init, c_init)
        #if mode == 'train':
        recon_loss, recon_acc, recon_preds = self.recon_loss(x, dec_h0, quantized_inputs, dict_codes, recon_type='sum')
        #else:
        #    recon_loss, recon_acc, recon_preds = self.recon_loss_test(x, dec_h0, quantized_inputs, dict_codes, recon_type='sum')

        # (batchsize)
        recon_loss = recon_loss.squeeze(1)
        vq_loss = vq_loss.squeeze(1)
        loss = recon_loss + vq_loss + kl_weight*kl_loss 
        return loss, recon_loss, vq_loss, recon_acc, quantized_inds, encoder_fhs, dict_codes, suffix_codes, recon_preds, kl_loss, logdet, selections

        
    def accuracy(self, output_logits, tgt, dict_codes):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = ((pred_tokens == tgt) * (tgt!=0)).sum().item()

        correct_predictions = []; wrong_predictions = []
        '''for i in range(batch_size):
            target  = ''.join([self.decoder.vocab.id2word(seq.item()) for seq in tgt[i]])
            pred  = ''.join([self.decoder.vocab.id2word(seq.item()) for seq in pred_tokens[i]])
            #target = target.replace('<p>','')
            #pred = pred.replace('<p>','')
            if target != pred:
                wrong_predictions.append('target: %s pred: %s, dict_code: %s' % (target, pred, dict_codes[i]))
            else:
                correct_predictions.append('target: %s pred: %s, dict_code: %s' % (target, pred, dict_codes[i]))'''
        return (acc, pred_tokens), (wrong_predictions, correct_predictions)

    