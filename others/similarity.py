import torch, argparse, matplotlib, random, sys
import pandas as pd  
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import VAE, LSTMEncoder, LSTMDecoder
from data import build_data, log_data
from collections import defaultdict
from numpy import save
from sklearn.datasets import fetch_openml
from utils import uniform_initializer
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
matplotlib.use('Agg')

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.ni = 512
args.enc_nh = 1024
args.dec_nh = 1024
args.nz = 32
args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.0
args.dec_dropout_out = 0.0
args.maxtrnsize = 1000 
args.maxvalsize = 8329 
args.maxtstsize = 8517
args.batchsize = 1
args.seq_to_no_pad = 'surface'
args.trndata = 'trmor_data/sigmorphon/2005/goldstdsample.tur'#'trmor_data/sigmorphon/2018task-1/turkish-train-high' 
args.valdata = args.trndata #'trmor_data/sigmorphon/2018task-1/turkish-dev' 
args.tstdata = args.trndata #'trmor_data/sigmorphon/2018task-1/turkish-test' 
args.surface_vocab_file = 'trmor_data/sigmorphon/2005/wordlist.tur' #'trmor_data/sigmorphon/2018task-1/turkish-train-high'#Turkish.bible.txt' 
args.task = 'sigmorphon2021task2/visualization'
args.bmodel = 'vae_morpho2005' 
numpoints = 52000
# DATA
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
trnbatches, valbatches, tstbatches = batches
surface_vocab  = vocab # fix this rebuilding vocab issue!
# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  
model = VAE(encoder, decoder, args)
model.encoder.is_reparam = False
if args.bmodel =='vae_morpho2005':
    args.basemodel = "models/vae/trmor_agg0_kls0.10_warm10_1_2611.pt"
    figname = 'z_space_vae_tsne.png'
elif args.bmodel =='ae_morpho2005':
    args.basemodel = "models/ae/trmor_agg0_kls0.00_warm10_1_2611.pt" #"trmor_agg0_kls0.00_warm10_1_1911.pt"
    figname = 'z_space_ae_tsne.png'
model.load_state_dict(torch.load(args.basemodel))
print('Model weights loaded from ... ', args.basemodel)
model.to(args.device)
model.eval()

# cossimilarity =  (a.b / ||a|| ||b||)  # see https://en.wikipedia.org/wiki/Cosine_similarity
def cos_similarity(a, b):
    a = a.cpu().detach()
    b = b.cpu().detach()
    result = dot(a, b.t())/(norm(a)*norm(b))
    return result

'''## Experiment1: 
## Similarities between encoder hidden states 
x = []; hx = []; zx = []; 
surf_strs = []; feat_strs = []
indices = list(range(len(trnbatches)))
random.seed(0); random.shuffle(indices)
f = open("ae_encoder_hs_similarities_morpho2005goldsample.txt", "w")
for i, idx in enumerate(indices):
    # (batchsize, t)
    #surf, feat = trnbatches[idx] 
    surf = trnbatches[idx] 
    # z: (batchsize, 1, nz), last_state: (1, 1, enc_nh), hidden_states: (1, t, enc_nh)
    z, KL, last_state, hidden_states = model.encode(surf, 1)
    _,tx, _ = hidden_states.shape
    surf_str = surface_vocab.decode_sentence(surf.t()) 
    f.write('\n---- ')
    f.write(''.join(surf_str)+'\n')
    f.write('\n')
    similarities = dict()
    # (1, enc_nh)
    fhs = hidden_states[:,tx-1, :]
    for t in range(tx):
        # (1, enc_nh)
        hs = hidden_states[:,t, :]
        similarities[''.join(surf_str[:t+1])] = cos_similarity(fhs, hs)[0]
    for k,v in sorted(similarities.items(), key=lambda item: item[1], reverse=True):
        f.write(str(k)+': '+str(v)+'\n')
        print((k,v))
f.close()'''


'''## Experiment2: 
## Similarities between mu vectors 
x = []; hx = []; zx = []; 
surf_strs = []; feat_strs = []
indices = list(range(len(trnbatches)))
random.seed(0); random.shuffle(indices)
f = open("vae_mu_similarities_morpho2005goldsample.txt", "w")
for i, idx in enumerate(indices):
    similarities = dict()
    # (batchsize, t)
    #surf, feat = trnbatches[idx] 
    surf = trnbatches[idx] 
    surf_str = surface_vocab.decode_sentence(surf.t()) 
    _, tx = surf.shape
    # fz: (batchsize, 1, nz), last_state: (1, 1, enc_nh), hidden_states: (1, t, enc_nh)
    fz, KL, last_state, hidden_states = model.encode(surf, 1)
    fz = fz.squeeze(0)
    eos = torch.tensor([2]).unsqueeze(0).to('cuda')
    for t in range(tx):
        _surf = surf[:,:t+1]
        if 2 not in _surf: # 2 is eos id
            _surf = torch.cat([_surf, eos], dim=1)
        z, KL, _, _ = model.encode(_surf, 1)
        z = z.squeeze(0)
        _surf_str = surface_vocab.decode_sentence(_surf.t()) 
        similarities[''.join(_surf_str)] = cos_similarity(fz, z)[0]
    f.write('\n---- ')
    f.write(''.join(surf_str)+'\n')
    f.write('\n')
    for k,v in sorted(similarities.items(), key=lambda item: item[1], reverse=True):
        f.write(str(k)+': '+str(v)+'\n')
        print((k,v))
f.close()'''



'''## Experiment3: 
## Similarities between decoder hidden states 
x = []; hx = []; zx = []; 
surf_strs = []; feat_strs = []
indices = list(range(len(trnbatches)))
random.seed(0); random.shuffle(indices)
f = open("ae_decoder_hs_similarities_morpho2005trn.txt", "w")
for i, idx in enumerate(indices):
    similarities = dict()
    # (batchsize, t)
    surf = trnbatches[idx] 
    surf_str = surface_vocab.decode_sentence(surf[:,:-1].t()) 
    f.write('\n---- ')
    f.write(''.join(surf_str)+'\n')
    f.write('\n')
    # z: (batchsize, 1, nz), last_state: (1, 1, enc_nh), hidden_states: (1, t, enc_nh)
    z , KL, last_state, hidden_states = model.encode(surf, 1)
    pred, _, decoder_hidden_states = model.decode(z.squeeze(0), 'greedy')
    pred = ['<s>']+pred[0] 
    ty,_,_ = decoder_hidden_states.shape
    # (1, dec_nh)
    fhs = decoder_hidden_states[ty-1,:, :]
    for t in range(ty):
        # (1, dec_nh)
        hs = decoder_hidden_states[t,:, :]
        similarities[''.join(pred[:t+1])] = cos_similarity(fhs, hs)[0]
    for k,v in sorted(similarities.items(), key=lambda item: item[1], reverse=True):
        f.write(str(k)+': '+str(v)+'\n')
        print((k,v))
f.close()'''


## Experiment4: 
## Eos probabilities at decoding
x = []; hx = []; zx = []; 
surf_strs = []; feat_strs = []
indices = list(range(len(trnbatches)))
random.seed(0); random.shuffle(indices)
f = open("vae_decoder_eos_probabilities_morpho2005goldsample.txt", "w")
for i, idx in enumerate(indices):
    probabilities = dict()
    # (batchsize, t)
    surf = trnbatches[idx] 
    surf_str = surface_vocab.decode_sentence(surf[:,:-1].t()) 
    f.write('\n---- ')
    f.write(''.join(surf_str)+'\n')
    f.write('\n')
    # z: (batchsize, 1, nz), last_state: (1, 1, enc_nh), hidden_states: (1, t, enc_nh)
    z , KL, last_state, hidden_states = model.encode(surf, 1)
    breakpoint()
    pred, _, decoder_hidden_states, eos_probs = model.decode(z.squeeze(0), 'greedy')
    
    pred = ['<s>']+pred[0] 
    ty,_,_ = decoder_hidden_states.shape
   
    for t in range(ty):
        probabilities[''.join(pred[:t+1])] = eos_probs[t]
    for k,v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        f.write(str(k)+': '+str(v)+'\n')
        print((k,v))
f.close()