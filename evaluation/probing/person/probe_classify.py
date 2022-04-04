# -----------------------------------------------------------
# Date:        2021/12/19
# Author:      Muge Kural
# Description: POS Classifier for trained probe 
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
from model.ae.ae import AE
from common.vocab import VocabEntry
from common.utils import *
from evaluation.probing.ae_probe import AE_Probe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
import sympy

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device         = 'cuda'
# personbook from probe
model_id = 'ae_for_vqvae_001_probe_person'
model_path, (surf_vocab, person_vocab)  = get_model_info(model_id) 

# data
with open(surf_vocab) as f:
    word2id = json.load(f)
    surf_vocab = VocabEntry(word2id)
with open(person_vocab) as f:
    word2id = json.load(f)
    person_vocab = VocabEntry(word2id)

# model
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 256; args.nz = 512; 
args.enc_nh = 512; args.dec_nh = 512
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.pretrained_model = AE(args, surf_vocab, model_init, emb_init)
args.nh = args.enc_nh
args.model = AE_Probe(args, person_vocab, model_init, emb_init)
args.model.load_state_dict(torch.load(model_path))
args.model.to(args.device)
args.model.eval()

# classify
def classify(word):
    data = [1]+ [surf_vocab[char] for char in word] + [2]
    x = torch.tensor(data).to('cuda').unsqueeze(0)
    sft = nn.Softmax(dim=2)
    # (1, 1, vocab_size)
    output_logits = args.model(x)
    probs = sft(output_logits)
    pred = torch.argmax(probs,2)[0][0].item()
    print(person_vocab.id2word(pred))
    return pred

#words = ['okursun', 'yazarsın', 'gülersin', 'seversin', 'çizersin', 'yazdın', 'çizdin', 'koştun']
#for word in words:


with open('evaluation/probing/root_concept/data/sosimple.new.seenroots.val.txt' , 'r') as reader:
    for line in reader:   
        word = line.split('\t')[0]
        model = args.model
        data = [1]+ [surf_vocab[char] for char in word] + [2]
        x = torch.tensor(data).to('cuda').unsqueeze(0)
        z, _ = model.encoder(x)
        #(1, hdim)
        z = z.squeeze(0)
        #(7, hdim)
        lw = model.linear.weight
        prod = (lw[1] * z).squeeze(0)
        maxind = torch.argmax(prod)
        pred_class = classify(word)
        #(1, hdim)
        class_vec = lw[pred_class].unsqueeze(0)
        matrix = torch.cat((class_vec, z),dim=0).detach().to('cpu')
        _, indexes = sympy.Matrix(matrix).T.rref(); 
        if len(indexes) != 2:
            print('NOT INDEPENDENT')

