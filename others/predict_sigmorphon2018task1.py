
import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from modules import VAE, LSTMEncoder, LSTMDecoder
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from data import build_data, log_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args):
    with open('tst_predictions.txt', "w") as fout:
        total_x_tokens = 0
        total_correct_tokens = 0
        total_pred_tokens = 0
        dsize = len(batches)
        indices = list(range(dsize))
        for i, idx in enumerate(indices):
            # (batchsize, t)
            surf, feat = batches[idx] 
            decoded_batch, _acc = args.model.reconstruct(feat, 'greedy', surf)
            if _acc != None:
                total_correct_tokens += _acc[0]
                total_x_tokens += _acc[1]
                total_pred_tokens += _acc[2]
            for sent in decoded_batch:
                fout.write("".join(sent) + "\n")
    recall = total_correct_tokens / total_x_tokens
    precision = total_correct_tokens/ total_pred_tokens
    F1 = 2 * (precision * recall) / (precision + recall)
    print('Recall: %.4f' % recall)
    print('Precision: %.4f' % precision)
    print('F1: %.4f' % F1)

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.maxtrnsize = 10000 
args.maxvalsize = 8329 
args.maxtstsize = 8517
args.task = 'sigmorphon2018task1/high-data'
args.trndata = 'trmor_data/sigmorphon/2018task-1/turkish-train-high' 
args.valdata = 'trmor_data/sigmorphon/2018task-1/turkish-dev' 
args.tstdata = 'trmor_data/sigmorphon/2018task-1/turkish-test' 
args.surface_vocab_file = 'trmor_data/sigmorphon/2018task-1/turkish-train-high' 

args.seq_to_no_pad = 'surface'
args.bmodel = 'vae_finetuned' 
args.batchsize = 1
rawdata, batches, vocab = build_data(args)
surface_vocab, feature_vocab = vocab
# MODEL
args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.0
args.dec_dropout_out = 0.0
args.ni = 512
args.enc_nh = 1024
args.dec_nh = 1024
args.nz = 32
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(feature_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  #feature_vocab for scratch preloading
args.model = VAE(encoder, decoder, args)
args.model.load_state_dict(torch.load('logs/sigmorphon2018task1/high-data/vae_original_2/10000_instances/100epochs.pt'))
args.model.to(args.device) 
_, _, tstbatches = batches

args.model.eval()
with torch.no_grad():
    test(tstbatches, 'tst', args)