import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

class TextVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, vocabsize, initrange=0.1, type='ae'):
        super().__init__()

        self.type = type
        self.embed_ae = nn.Embedding(vocabsize, hidden_dim)
        self.embed_s2s = nn.Embedding(vocabsize, hidden_dim)
      
        self.E_ae = nn.LSTM(hidden_dim, hidden_dim, 1, dropout=0.0, bidirectional=True)
        self.E_s2s = nn.LSTM(hidden_dim, hidden_dim, 1, dropout=0.0, bidirectional=True)

        if 'vae' in self.type:
            #self.E_s2s = nn.LSTM(hidden_dim, 16, 1, dropout=0.0, bidirectional=True)
            self.h2_mu = nn.Linear(hidden_dim*2, latent_dim)
            self.h2_logvar = nn.Linear(hidden_dim*2, latent_dim)
            self.trans_linear = nn.Linear(latent_dim, hidden_dim*2)
            self.G = nn.LSTM(hidden_dim+latent_dim, hidden_dim*2, 1, dropout=0.0)
        else:
            self.G = nn.LSTM(hidden_dim, hidden_dim*2, 1, dropout=0.0)

        self.proj = nn.Linear(hidden_dim*2, vocabsize)

        self.embed_s2s.weight.data.uniform_(-initrange, initrange)
        self.embed_ae.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)
        
        self.drop = nn.Dropout(0.5)
        self.padidx = 0
        loc = torch.zeros(latent_dim, device=device)
        scale = torch.ones(latent_dim, device=device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        
    def encode_s2s(self, input):
        input = self.drop(self.embed_ae(input))
        _, (ht, _) = self.E_s2s(input)
        hes = torch.cat([ht[-2], ht[-1]], 1) 
        return hes 
    
    def encode_ae(self, input):
        if 'ft' in self.type:
            input = self.embed_ae(input)
        else:
            input = self.embed_ae(input) #self.drop(self.embed_ae(input))
        _, (ht, _) = self.E_ae(input)
        hes = torch.cat([ht[-2], ht[-1]], 1) 
        return hes 

    def encode_vae(self, input):
        if 'ft' in self.type:
            input = self.embed_ae(input)
        else:
            input = self.drop(self.embed_ae(input))
        _, (ht, _) = self.E_ae(input)
        hes = torch.cat([ht[-2], ht[-1]], 1) 
        return self.h2_mu(hes), self.h2_logvar(hes)
    
    
    def decode_teacher_forcing(self, z, target_inputs, target_outputs):
        if 'ft' in self.type: 
            word_embed = self.embed_ae(target_inputs)
        else:
            word_embed = self.drop(self.embed_ae(target_inputs))
        if 'vae' in self.type:
            _z = z.repeat(word_embed.size(0),1,1)
            word_embed = torch.cat((word_embed, _z), -1)
            c = self.trans_linear(z).unsqueeze(0)
        else:
            c = z.unsqueeze(0) 

        h = torch.tanh(c)
        decoder_hidden = (h,c)
        output, hidden = self.G(word_embed, decoder_hidden)
        logits = self.proj(output.view(-1, output.size(-1)))
        logits = logits.view(output.size(0), output.size(1), -1) 
        rec_loss = (F.cross_entropy(logits.view(-1, logits.size(-1)), target_outputs.flatten(), ignore_index=self.padidx, reduction='none').view(target_outputs.size()).sum(dim=0)).mean()
        acc = self.accuracy(logits, target_outputs)
        return rec_loss, acc

    def decode_input_feeding(self, z, vocab, mode, target_outputs=None):
        B = z.size(0)
        if target_outputs is not None:
            target_length = target_outputs.size(0)
        else:
            target_length = 30
        SOS_TOKEN = 1
        decoder_input = torch.ones((1,B), device=device).long()

        if 'vae' in self.type:
            c = self.trans_linear(z).unsqueeze(0) #1,B,H
        else:
            c = z.unsqueeze(0)
            
        h = torch.tanh(c)
        decoder_hidden = (h,c)
        sft = nn.Softmax(2)
        preds = []
        losses = []
        for di in range(target_length):
            if 'ft' in self.type:
                word_embed =  self.embed_ae(decoder_input) #1,B,H
            else:
                word_embed =  self.drop(self.embed_ae(decoder_input)) #1,B,H
            if 'vae' in self.type:
                word_embed =  torch.cat((word_embed, z.unsqueeze(0)), -1)

            output, decoder_hidden = self.G(word_embed, decoder_hidden)
            logits = self.proj(output.view(-1, output.size(-1))).unsqueeze(0)
            decoder_input = torch.argmax(sft(logits),2) # T,B
            if target_outputs is not None:
                losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), target_outputs[di], ignore_index = self.padidx, reduction='none'))
            preds.append(decoder_input.squeeze(0))
        preds = torch.stack(preds).to(device)
        rec_loss = None
        acc = None
        if target_outputs is not None:
            rec_loss =  torch.stack(losses).sum(dim=0).mean() 
            acc = (((preds == target_outputs) * (target_outputs != self.padidx)).sum() / (target_outputs != self.padidx).sum()).item()
        return rec_loss, acc, preds
   
    def loss(self, input, targets, vocab, mode, teac_forc, beta=1):
        
        total_loss = torch.tensor(0).to(device).float(); align_loss = torch.tensor(0).to(device); kl_loss = torch.tensor(0).to(device); 
        l1loss = nn.L1Loss()
        
        if self.type == 'vae':
            mu, logvar = self.encode_vae(targets)
            z = reparameterize(mu,logvar)
            kl_loss = loss_kl(mu, logvar) * beta
            total_loss += kl_loss

        elif self.type == 's2s':
            z_s2s = self.encode_s2s(input)
            z = self.encode_ae(targets)
            align_loss = l1loss(z, z_s2s)
            total_loss += align_loss
                    
        elif self.type == 'ae':
            z = self.encode_ae(targets)

        elif 'ft' in self.type:
            z = self.encode_s2s(input)
            if 'vae' in self.type:
                mu, logvar = self.encode_vae(targets)
                z_vae = reparameterize(mu,logvar)
                align_loss = l1loss(z, z_vae)
                total_loss += align_loss
            elif 'ae' in self.type:
                z_ae = self.encode_ae(targets)
                align_loss = l1loss(z, z_ae)
                total_loss += align_loss
        target_inputs  =  targets[:-1,:]
        target_outputs =  targets[1:,:]
        
        if teac_forc:
            rec_loss, acc = self.decode_teacher_forcing(z, target_inputs, target_outputs)
            #_, acc = self.decode_teacher_forcing(z_s2s, target_inputs, target_outputs)
        else:
            rec_loss, acc, _ = self.decode_input_feeding(z, vocab, mode, target_outputs=target_outputs)
            #_, acc, _ = self.decode_input_feeding(z_s2s, vocab, mode, target_outputs=target_outputs)

            
        if 'ft' not in self.type or 'ft-vae' in self.type:
            total_loss += rec_loss
      
            
        return total_loss, rec_loss, align_loss, kl_loss, acc
    
    def pred(self, input, vocab):
        mu, logvar = self.encode(input)
        #kl_loss = loss_kl(mu, logvar) 
        z = mu #reparameterize(mu, logvar)
        rec_loss, acc, _ = self.decode_input_feeding(z, vocab, 'tst')
        return rec_loss, acc, _ #(kl_anneal *kl_loss) + rec_loss

    
    def accuracy(self, logits, targets, is_dev=False):
        T, B = targets.size()
        sft = nn.Softmax(dim=2)
        pred_tokens = torch.argmax(sft(logits),2) # T,B
        acc = ((pred_tokens == targets) * (targets != self.padidx)).sum() / (targets != self.padidx).sum()
        return acc.item()

    def sample_from_prior(self, z=None):
        if z is None:
            z = self.prior.sample((1,))
        SOS_TOKEN= 1
        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
        c = self.trans_linear(z).unsqueeze(0) #1,B,H
        h = torch.tanh(c)
        decoder_hidden = (h,c)
        sft = nn.Softmax(2)
        preds = []
        MAX_LENGTH=30
        for di in range(MAX_LENGTH):
            word_embed =  self.embed_ae(decoder_input) #1,B,H
            word_embed =  torch.cat((word_embed, z.unsqueeze(0)), -1)
            output, decoder_hidden = self.G(word_embed, decoder_hidden)
            logits = self.proj(output.view(-1, output.size(-1))).unsqueeze(0)
            decoder_input = torch.argmax(sft(logits),2) # T,B
            preds.append(decoder_input.squeeze(0))
            if decoder_input.item() == 2:
                break
        return preds, z  

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

    
    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))
        Returns: Float
        """

        # [x_batch, nz]
        # mu, logvar = self.forward(x)
        mu, logvar = self.encode_vae(x)
        
        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi)- 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
    
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size)

#     def forward(self, input):
#         T,B = input.size()
#         embedded = self.embedding(input).view(T, B, -1)
#         output = embedded
#         output, (h0, c0) = self.lstm(output)
#         return output, (h0, c0) # T,B,H, and 1,B,H


# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=2)

#     def forward(self, input, h0, c0):
#         T, B = input.size()
#         output = self.embedding(input).view(T, B, -1) # T,B,H
#         output = F.relu(output)
#         output, (h,c) = self.lstm(output, (h0, c0)) 
#         output = self.softmax(self.out(output)) # T,B,vocabsize
#         preds = output.argmax(dim=2) # T,B
#         return output, (h,c), preds

    
# def loss(input_tensor, target_tensor, encoder, decoder, criterion, vocab=None, is_report=False):
#     T,B = input_tensor.size()
#     target_inputs  =  target_tensor[:-1,:]
#     target_outputs =  target_tensor[1:,:]
#     input_length = input_tensor.size(0)
#     target_length = target_outputs.size(0)
#     _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
#     decoder_output, (h,c), preds = decoder(target_inputs, encoder_hidden, encoder_cell) # preds: T,B
#     acc = ((preds == target_outputs) * (target_outputs != 0)).sum() / (target_outputs != 0).sum()
#     if is_report:
#         tokens = []
#         for t in preds[:,0]:
#             token = vocab.idx2word[t]
#             if token == '<eos>':
#                 break
#             else:
#                 tokens.append(token)
#         #print(tokens)
#     loss = criterion(decoder_output.view(-1, decoder_output.size(2)), target_outputs.view(-1))
#     return loss / (target_length-1), acc.item()


# def test(input_tensor, target_tensor, encoder, decoder, criterion, vocab):
#     T,B = input_tensor.size()
#     target_outputs =  target_tensor[1:,:]
#     input_length = input_tensor.size(0)
#     target_length = target_outputs.size(0)
#     _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
#     h, c = encoder_hidden, encoder_cell
#     SOS_TOKEN= vocab.word2idx['<go>']
#     EOS_TOKEN= vocab.word2idx['<eos>']
#     decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
#     loss = 0
#     preds = []
#     for di in range(target_length):
#         decoder_output, (h,c), decoder_input = decoder(decoder_input, h, c)
#         preds.append(decoder_input.item())
#         loss += criterion(decoder_output.view(-1, decoder_output.size(2)), target_outputs[di])
#     preds = torch.tensor(preds).to(device)
#     acc = (preds == target_outputs.flatten()).sum() /  (target_outputs != 0).sum()
#     return loss / (target_length-1), acc.item()
    

# def pred(input_tensor, encoder, decoder, vocab):
#     _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
#     h, c = encoder_hidden, encoder_cell
#     SOS_TOKEN = vocab.word2idx['<go>']
#     EOS_TOKEN = vocab.word2idx['<eos>']
#     decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
#     preds = []
#     for di in range(50):
#         decoder_output, (h,c), decoder_input = decoder(decoder_input, h, c)
#         preds.append(decoder_input.item())
#         if decoder_input.item() == EOS_TOKEN:
#             break
#     tokens = []
#     for t in preds:
#         token = vocab.idx2word[t]
#         if token == '<eos>':
#             break
#         else:
#             tokens.append(token)
#     return tokens


