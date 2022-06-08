# ref: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
import torch, json
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

        self.linear = nn.Linear(args.enc_nh*2, 2*args.nz, bias=False)

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
       
        mean, logvar = self.linear(last_state).chunk(2, -1)
        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        fwd = fwd.permute(1,0,2)
        bck = bck.permute(1,0,2)

        last_cell = last_cell.permute(1,0,2)
        return last_state, last_cell, None, fwd, bck, mean.squeeze(0), logvar.squeeze(0)

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
        return quantized_latents.contiguous(), vq_loss, encoding_inds.t()#, torch.topk(dist,5, largest=False)[1]

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
        root_z, suffix_z = z
        batch_size, _, _ = root_z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)
        z_ = suffix_z.expand(batch_size, seq_len, self.incat)# 64
        # (batch_size, seq_len, ni + nz)
        word_embed = torch.cat((word_embed, z_), -1)
        
        # (1, batch_size, nz)
        root_z = root_z.permute((1,0,2))
        # (1, batch_size, dec_nh)
        c_init = root_z
        #c_init = self.trans_linear(z).unsqueeze(0)
        h_init = torch.tanh(c_init)
        output, _ = self.lstm(word_embed, (h_init, c_init))
        output = self.dropout_out(output)

        #z_ = suffix_2_z.expand(batch_size, seq_len, 128)
        # (batch_size, seq_len, ni + nz)
        #output = torch.cat((output, z_), -1)

        # (batch_size, seq_len, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits

class VQVAE(nn.Module):

    def __init__(self,
                args,
                vocab,
                model_init,
                emb_init,
                dict_assemble_type='concat',
                 **kwargs) -> None:
        super(VQVAE, self).__init__()
        self.nz = args.nz


        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        self.kl_weight = 0.2

        self.encoder = VQVAE_Encoder(args, vocab, model_init, emb_init) 
        self.encoder_emb_dim = args.embedding_dim 
        self.num_dicts = args.num_dicts
        self.orddict_emb_num = args.orddict_emb_num
        self.dict_assemble_type = dict_assemble_type
        
        #assert (dict_assemble_type=='concat' or (dict_assemble_type =='sum' and self.rootdict_emb_dim == self.encoder_emb_dim)), "If dict assemble type is sum, dict embedding dim should be equal to encoder emb dim"

        if dict_assemble_type =='concat':
            self.orddict_emb_dim = int((self.encoder_emb_dim-self.rootdict_emb_dim)/(self.num_dicts))
        elif dict_assemble_type == 'sum':
            self.orddict_emb_dim = self.rootdict_emb_dim
        else:
            self.orddict_emb_dim   = int(args.incat/self.num_dicts)
            
        self.beta = args.beta

        #self.linear_root = nn.Linear(self.encoder_emb_dim,  args.dec_nh, bias=True)
    
        #other
        #self.ord_linears = nn.ModuleList([])
        self.ord_vq_layers = nn.ModuleList([])

        #for i in range(1):#(self.num_dicts-1):
        for i in range(self.num_dicts):
            #self.ord_linears.append(nn.Linear(self.encoder_emb_dim, self.orddict_emb_dim, bias=True))
            self.ord_vq_layers.append(VectorQuantizer(self.orddict_emb_num,
                                        self.orddict_emb_dim,
                                        self.beta))
        self.decoder = VQVAE_Decoder(args, vocab, model_init, emb_init) 
    

    def vq_loss(self,x, epc):
        # fhs: (B,1,hdim)

        #fhs, _, _, fwd_fhs, bck_fhs = self.encoder(x)
        fhs, _, _, fwd_fhs, bck_fhs, mu, logvar = self.encoder(x)
        _root_fhs = self.reparameterize(mu, logvar)
        # (batchsize)
        kl_loss = self.kl_loss(mu,logvar)
        #fhs, _, _ = self.encoder(x)
        
        #fhs = fwd_fhs
        vq_vectors = []; vq_losses = []; vq_inds = []
        # quantized_input: (B, 1, rootdict_emb_dim)
        #_root_fhs =  self.linear_root(fwd_fhs)
        #_root_fhs = torch.sigmoid(_root_fhs)
        vq_vectors.append(_root_fhs)

        # quantize thru ord dicts
        i=0
        #for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
        for vq_layer in self.ord_vq_layers:
            #_fhs =  linear(fwd_fhs)
            # quantized_input: (B, 1, orddict_emb_dim)
            quantized_input, vq_loss, quantized_inds = vq_layer(bck_fhs[:,:,i*self.orddict_emb_dim:(i+1)*self.orddict_emb_dim],epc)
            #quantized_input, vq_loss, quantized_inds = vq_layer(_fhs,epc)
            vq_vectors.append(quantized_input)
            vq_losses.append(vq_loss)
            vq_inds.append(quantized_inds)
            i+=1
        
        '''sufv   = torch.cat(vq_vectors[1:],dim=1)
        logdet = 0
        for i in range(sufv.shape[0]):
            mtx =  torch.cov(sufv[i].t())
            logdet += torch.log(torch.det(mtx)+1e-6)
            #print(logdet)'''

        #b,t,128
        #ins = torch.cat(vq_vectors[1:],dim=1)
        # (batch_size, seq_len-1, args.ni)
        # last_state: 1,128,256
        #output, (last_state, last_cell) = self.suffix_lstm(ins)
        #last_state = last_state.squeeze(0).unsqueeze(1)

        if self.dict_assemble_type == 'sum':
            for i in range(0, len(vq_vectors)):
                vq_vectors[0] += vq_vectors[i]
            vq_vectors = vq_vectors[0]
        elif self.dict_assemble_type == 'concat':
            # concat quantized vectors
            vq_vectors = torch.cat(vq_vectors,dim=2)
        else:
            #vq_vectors = (vq_vectors[0], last_state)
            #for i in range(2, len(vq_vectors)):
            #    vq_vectors[1] += vq_vectors[i]
            #vq_vectors = (vq_vectors[0], vq_vectors[1]) #vq_vectors[1])
            vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2)) 

     
        # (batchsize, numdicts)
        vq_loss =  torch.cat(vq_losses,dim=1)
        # (batchsize)
        vq_loss = vq_loss.sum(-1)
        # (batchsize,1)
        vq_loss = vq_loss.unsqueeze(1)

        dict_codes = []; suffix_codes = []
        for i in range(vq_loss.shape[0]):
            dict_code = str((vq_inds[0][0][i]).item())
            suffix_code = "" #str((vq_inds[0][0][i]).item())
            for j in range(0, len(vq_inds)):
                dict_code   += '-' + str((vq_inds[j][0][i]).item())
                suffix_code += '-' + str((vq_inds[j][0][i]).item()) 
            dict_codes.append(dict_code)
            suffix_codes.append(suffix_code)
        return vq_vectors, vq_loss, vq_inds,  fhs, dict_codes, suffix_codes, 0, kl_loss

    def recon_loss(self, x, quantized_z, dict_codes=None, recon_type='avg'):
        # remove end symbol
        src = x[:, :-1]
        # remove start symbol
        tgt = x[:, 1:]        
        batch_size, seq_len = src.size()
        # (batch_size, seq_len, vocab_size)
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
        recon_acc, recon_preds  = self.accuracy(output_logits, tgt, dict_codes)
        return recon_loss, recon_acc, recon_preds

    def kl_loss(self, mu, logvar):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return KL

    def loss(self, x: Tensor, epc, **kwargs) -> List[Tensor]:
        # x: (B,T)
        # quantized_inputs: (B, 1, hdim)
        quantized_inputs, vq_loss, quantized_inds, encoder_fhs, dict_codes, suffix_codes, logdet, kl_loss = self.vq_loss(x, epc)
        recon_loss, recon_acc, recon_preds = self.recon_loss(x, quantized_inputs, dict_codes, recon_type='sum')
        # (batchsize)
        recon_loss = recon_loss.squeeze(1)
        vq_loss = vq_loss.squeeze(1)
        loss = recon_loss + vq_loss + (self.kl_weight * kl_loss)
        return loss, recon_loss, vq_loss, kl_loss, recon_acc, quantized_inds, encoder_fhs, dict_codes, suffix_codes, recon_preds, logdet

        
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

    def log_probability_w_reinflection(self, x, root_fhs=None, recon_type='sum'):
        z0, _, _ = self.vq_layer_root(root_fhs,0)
        probs = [] 
        for j1 in range( self.orddict_emb_num):
            for j2 in range( self.orddict_emb_num):
                for j3 in range( self.orddict_emb_num):
                    for j4 in range( self.orddict_emb_num):
                        for j5 in range( self.orddict_emb_num):
                            for j6 in range( self.orddict_emb_num):
                                for j7 in range( self.orddict_emb_num):
                                    for j8 in range( self.orddict_emb_num):
                                        vq_vectors = []
                                        z1 = self.ord_vq_layers[0].embedding.weight[j1].unsqueeze(0).unsqueeze(0)
                                        z2 = self.ord_vq_layers[1].embedding.weight[j2].unsqueeze(0).unsqueeze(0)
                                        z3 = self.ord_vq_layers[2].embedding.weight[j3].unsqueeze(0).unsqueeze(0)
                                        z4 = self.ord_vq_layers[3].embedding.weight[j4].unsqueeze(0).unsqueeze(0)
                                        z5 = self.ord_vq_layers[4].embedding.weight[j5].unsqueeze(0).unsqueeze(0)
                                        z6 = self.ord_vq_layers[5].embedding.weight[j6].unsqueeze(0).unsqueeze(0)
                                        z7 = self.ord_vq_layers[6].embedding.weight[j7].unsqueeze(0).unsqueeze(0)
                                        z8 = self.ord_vq_layers[7].embedding.weight[j8].unsqueeze(0).unsqueeze(0)
                                        vq_vectors.append(z0)
                                        vq_vectors.append(z1)
                                        vq_vectors.append(z2)
                                        vq_vectors.append(z3)
                                        vq_vectors.append(z4)
                                        vq_vectors.append(z5)
                                        vq_vectors.append(z6)
                                        vq_vectors.append(z7)
                                        vq_vectors.append(z8)
                                        vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
                                        recon_loss, _, _ = self.recon_loss(x, vq_vectors, recon_type=recon_type)
                                        probs.append(-recon_loss.item())
        return sum(probs)/len(probs)

    def log_probability_w_recon(self, x, quantized_input_root, recon_type='sum'):
        vq_vectors = []
        fhs, _, _ = self.encoder(x)

        if quantized_input_root == None:
            quantized_input_root, vq_loss, _ = self.vq_layer_root(fhs,0)
        vq_vectors.append(quantized_input_root)
        # quantize thru ord dicts
        for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
            _fhs =  linear(fhs)
            quantized_input, vq_loss, _ = vq_layer(_fhs,0)
            vq_vectors.append(quantized_input)
        vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
        recon_loss, _, _ = self.recon_loss(x, vq_vectors, recon_type=recon_type)
        return (-recon_loss).item()

    '''def log_probability(self, x, root_fhs=None, recon_type='sum'):
        with open('model/vqvae/results/training/50000_instances/1x10000_3x8/0_cluster_usage.json','r') as reader:
            rootprobs = dict()
            for line in reader:
                line = line.strip()
                for k,v in json.loads(line).items():
                    rootprobs[k] = v/10000
        with open('model/vqvae/results/training/50000_instances/1x10000_3x8/4_cluster_usage_test.json','r') as reader:
            zprobs = dict()
            for line in reader:
                line = line.strip()
                for k,v in json.loads(line).items():
                    zprobs[k] = v/50000
        vq_inds =[];  vq_vectors = []
        fhs, _, _ = self.encoder(x)
        quantized_input_root, vq_loss, quantized_inds_root = self.vq_layer_root(fhs,0)
        # quantize thru ord dicts
        for linear, vq_layer in zip(self.ord_linears, self.ord_vq_layers):
            _fhs =  linear(fhs)
            # quantized_input: (B, 1, orddict_emb_dim)
            quantized_input, vq_loss, quantized_inds = vq_layer(_fhs,0)
            vq_vectors.append(quantized_input)
            vq_inds.append(str(quantized_inds.item()))
        vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
        recon_loss, _, _ = self.recon_loss(x, vq_vectors, dict_codes=None, recon_type=recon_type)
        vq_code =  '-'.join(vq_inds) 
        vq_code_prob = 0; root_code_prob = 0
        if vq_code in zprobs:
            vq_code_prob = zprobs[vq_code]
        if str(quantized_inds_root.item()) in rootprobs:
            root_code_prob =  rootprobs[str(quantized_inds_root.item())]
        prob = torch.exp(-recon_loss).item() * vq_code_prob  * root_code_prob
        return prob
        probs = [] 
        #for j0 in range( self.rootdict_emb_num):
        for j1 in range( self.orddict_emb_num):
            for j2 in range( self.orddict_emb_num):
                for j3 in range( self.orddict_emb_num):
                    vq_vectors = []
                    vq_vectors.append(quantized_input_root)
                    #z0 = self.vq_layer_root.embedding.weight[j0].unsqueeze(0).unsqueeze(0)
                    #vq_vectors.append(z0)
                    z1 = self.ord_vq_layers[0].embedding.weight[j1].unsqueeze(0).unsqueeze(0)
                    z2 = self.ord_vq_layers[1].embedding.weight[j2].unsqueeze(0).unsqueeze(0)
                    z3 = self.ord_vq_layers[2].embedding.weight[j3].unsqueeze(0).unsqueeze(0)
                    vq_vectors.append(z1)
                    vq_vectors.append(z2)
                    vq_vectors.append(z3)
                    vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
                    recon_loss, _, _ = self.recon_loss(x, vq_vectors, recon_type=recon_type)
                    vq_code =  str(j1)+'-'+str(j2)+'-'+str(j3)
                    vq_code_prob = 0; root_code_prob = 0
                    if vq_code in zprobs:
                        vq_code_prob = zprobs[vq_code]
                    if str(quantized_inds_root.item()) in rootprobs:
                        root_code_prob =  rootprobs[str(quantized_inds_root.item())]
                    prob = torch.exp(-recon_loss).item() * vq_code_prob  * root_code_prob
                    probs.append(prob)
        return sum(probs)'''
