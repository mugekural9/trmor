# -----------------------------------------------------------
# Date:        2022/03/09 
# Author:      Muge Kural
# Description: Trainer of type-level inflection (feature forms to surface forms), saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from common.utils import *
from data.data import build_data
from common.vocab import VocabEntry

matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    for epc in range(args.epochs):
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            # (batchsize, t)
            surf, feat = trnbatches[idx]
            # TODO...            

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'

# training
args.seq_to_no_pad = 'feature'
args.batchsize = 64
args.epochs = 100

# data
args.trndata = 'evaluation/type_level_inflection/data/surf.uniquesurfs.trn.txt' 
args.valdata = 'evaluation/type_level_inflection/data/surf.uniquesurfs.val.txt' 
args.tstdata = args.valdata
args.surface_vocab_file = args.trndata
args.maxtrnsize = 57769; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab = build_data(args)
_, feature_vocab  = vocab
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(tstdata)

train(batches, args)
# TODO model and training
