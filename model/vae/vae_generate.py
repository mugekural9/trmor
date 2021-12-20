# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Word generator through sampling from prior for trained VAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vae import VAE
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_path = 'model/vae/results/50000_instances/30epochs.pt'
model_vocab = 'model/vae/results/50000_instances/surf_vocab.json'

# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(model_vocab) as f:
    word2id = json.load(f)
    vocab = VocabEntry(word2id)
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 512; args.nz = 32; 
args.enc_nh = 1024; args.dec_nh = 1024
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
model = VAE(args, vocab, model_init, emb_init)

# load model weights
model.load_state_dict(torch.load(model_path))
model.eval()



def sample(model):
    bosid = vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    # (1,1,nz)
    z = model.prior.sample((1,)).to('cpu:0').unsqueeze(0)
    # (1,1,dec_nh)
    c_init = model.decoder.trans_linear(z)
    h_init = torch.tanh(c_init)
    decoder_hidden = (c_init, h_init)
    copied = []; i = 0; max_length = 20
    while i < max_length:
        i +=1
        # (1,1,ni)
        word_embed = model.decoder.embed(torch.tensor([input]).unsqueeze(0))
        word_embed = torch.cat((word_embed, z), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits)) # sample
        char = model.decoder.vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            print(''.join(copied))
            break

# generate word
sample(model)