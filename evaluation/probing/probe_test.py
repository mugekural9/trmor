import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
from modules import VAE, LSTMEncoder, LSTMDecoder
from data.vocab import VocabEntry
from utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

print(os.getcwd())
def build_model(args, surface_vocab, surfpos_vocab):
    # MODEL
    # must be that way to load pretrained ae and vae
    args.ni = 512
    args.enc_nh = 1024
    args.dec_nh = 1024
    args.nz = 32
    args.enc_dropout_in = 0.0
    args.dec_dropout_in = 0.0
    args.dec_dropout_out = 0.0
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
    decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init) 
    args.model = VAE(encoder, decoder, args)
    args.model.encoder.is_reparam = False
    args.model.encoder.linear = None
    args.model.decoder = None
    args.model.probe_linear = nn.Linear(args.enc_nh, len(surfpos_vocab), bias=False)
    if args.bmodel == 'vae_original_probe':
        args.model.load_state_dict(torch.load('probe/surf2surfpos/vae_original/14000_instances/100epochs.pt'))
    if args.bmodel == 'ae_original_probe':
        args.model.load_state_dict(torch.load('probe/surf2surfpos/ae_original/14000_instances/100epochs.pt'))   
    args.model.to(args.device)  
    return args.model

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.bidirectional = False;
args.bmodel =  'ae_original_probe' 



with open('probe/surf2surfpos/ae_original/14000_instances/feat_vocab.json') as f:
    word2id = json.load(f)
    feat_vocab = VocabEntry(word2id)

with open('probe/surf2surfpos/ae_original/14000_instances/polar_vocab.json') as f:
    word2id = json.load(f)
    polar_vocab = VocabEntry(word2id)

with open('probe/surf2surfpos/ae_original/14000_instances/pos_vocab.json') as f:
    word2id = json.load(f)
    pos_vocab = VocabEntry(word2id)

with open('probe/surf2surfpos/ae_original/14000_instances/surf_vocab.json') as f:
    word2id = json.load(f)
    surface_vocab = VocabEntry(word2id)

with open('probe/surf2surfpos/ae_original/14000_instances/surfpos_vocab.json') as f:
    word2id = json.load(f)
    surfpos_vocab = VocabEntry(word2id)

with open('probe/surf2surfpos/ae_original/14000_instances/tense_vocab.json') as f:
    word2id = json.load(f)
    tense_vocab = VocabEntry(word2id)


# model
args.model = build_model(args, surface_vocab, surfpos_vocab)
args.model.eval()


# predict
def predict(word):
    data = [1]+ [surface_vocab[char] for char in word] + [2]
    x = torch.tensor(data).to('cuda').unsqueeze(0)
    sft = nn.Softmax(dim=2)
    z, _, last_states = args.model.encode(x, 1)
    # (1, batchsize, vocab_size)
    output_logits = args.model.probe_linear(last_states) 
    print(output_logits)
    probs = sft(output_logits)
    pred_tokens = torch.argmax(probs,2)
    pred = pred_tokens[0][0].item()
    print(surfpos_vocab.id2word(pred))

predict('evlerin')