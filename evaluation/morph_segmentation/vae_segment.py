# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Morpheme segmentation heuristics for trained VAE model
# -----------------------------------------------------------

import string
from common.vocab import VocabEntry
from model.vae.vae import VAE
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
import numpy as np
from data.data import build_data
from collections import OrderedDict

# heur2: detects morpheme boundary if: 
#        (1) the current likelihood(ll) exceeds prev and next ll OR current ll increase excesses prev inc
def heur_prev_mid_next_and_prevnext_exceed(logps, eps):
    morphemes = []
    prev_word = ''
    logps = [(k,v) for k,v in logps.items()]
    for i in range(1,len(logps)-1):
        prev = logps[i-1][1]
        cur = logps[i]
        nex = logps[i+1][1]
        if (i>0 and (cur[1] > (prev + nex)/2)  and len(cur[0])>2) or (cur[1] > prev + eps and cur[1] > nex and len(cur[0])>2): 
            morph = cur[0][-(len(cur[0])-len(prev_word)):]
            morphemes.append(morph)
            prev_word = cur[0]
    # add full word
    morph = logps[-1][0][-(len(logps[-1][0])-len(prev_word)):]
    morphemes.append(morph)
    return morphemes


# heur1: detects morpheme boundary if: 
#        (1) the current likelihood(ll) exceeds prev and next ll
def heur_prev_mid_next(logps, eps):
    morphemes = []
    prev_word = ''
    logps = [(k,v) for k,v in logps.items()]
    for i in range(1,len(logps)-1):
        if i==1:
            prev_of_prev = 0
        else:
            prev_of_prev =logps[i-2][1]
        prev = logps[i-1][1]
        cur = logps[i]
        nex = logps[i+1][1]
        if (cur[1] > prev + eps and cur[1] > nex and len(cur[0])>2): 
            morph = cur[0][-(len(cur[0])-len(prev_word)):]
            morphemes.append(morph)
            prev_word = cur[0]
    # add full word
    morph = logps[-1][0][-(len(logps[-1][0])-len(prev_word)):]
    morphemes.append(morph)
    return morphemes

# returns log likelihood of given word and its subwords
def get_logps(args, word, data, from_file=False):
    if from_file:
        with open(args.fprob, 'r') as json_file:
            logps = json.load(json_file)
            return logps[word]
    else:    
        with torch.no_grad():
            logps = dict()
            mu, logvar, _ = args.model.encoder(data)
            param = (mu,logvar,_)
            z = args.model.reparameterize(mu, logvar, args.nsamples)
            logpx = args.model.nll_iw(data, args.nsamples, z, param, args.recon_type)
            logpx = torch.mean(logpx).item()
            logps[word] = logpx
            # loop through word's subwords 
            for i in range(len(data[0])-2, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                if args.sample_type == 'subword_given':# sample z from subword 
                    mu, logvar, _ = args.model.encoder(subdata.unsqueeze(0))
                    param = (mu,logvar,_)
                    z = args.model.reparameterize(mu, logvar, args.nsamples)
                    logpx = args.model.nll_iw(subdata.unsqueeze(0), args.nsamples, z, param, args.recon_type)
                    logpx = torch.mean(logpx).item()
                else:
                    # sample z from full word (i.e. word_given)
                    logpx = args.model.nll_iw(subdata.unsqueeze(0), args.nsamples, z, param, args.recon_type)
                    logpx = torch.mean(logpx).item()
                logps[subword] = logpx #"{:.3f}".format(logpx)
        logps = dict(reversed(list(logps.items())))
        return logps


# returns log likelihood of given word subwords and all
def get_logps_matrix(args, word, data, from_file=False):
    if from_file:
        with open(args.fprob, 'r') as json_file:
            logps = json.load(json_file)
            return logps[word]
    else:    
        with torch.no_grad():
            matrix = dict()
            # loop through word's subwords 
            for i in range(len(data[0])-1, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                logps = dict()
                mu, logvar, _ = args.model.encoder(subdata.unsqueeze(0))
                param = (mu,logvar,_)
                z = args.model.reparameterize(mu, logvar, args.nsamples)
                logpx = args.model.nll_iw(data, args.nsamples, z, param, args.recon_type)
                logpx = torch.mean(logpx).item()
                logps[subword] = logpx
                # loop through word's subwords 
                for i in range(len(data[0])-1, 1, -1):
                    eos  = torch.tensor([2]).to(args.device)
                    subsubdata = torch.cat([data[0][:i], eos])
                    subsubword = ''.join(args.vocab.decode_sentence(subsubdata[1:-1]))
                    # sample z from full word (i.e. word_given)
                    logpx = args.model.nll_iw(subsubdata.unsqueeze(0), args.nsamples, z, param, args.recon_type)
                    logpx = torch.mean(logpx).item()
                    logps[subsubword] = logpx
                logps =  dict(sorted(logps.items())) #dict(reversed(list(logps.items())))
                matrix[subword] = logps
        return matrix


def config(args):
     # CONFIG
    args.device = 'cuda'
    #model_id = 'vae_segm_05_20'
    model_path, model_vocab  = get_model_info(args.model_id)
    # heuristic
    args.heur_type = 'prev_mid_next'; args.eps = 0.0
    args.nsamples = 10000
    # (a) avg: averages ll over word tokens, (b) sum: adds ll over word tokens
    #args.recon_type = 'sum' 
    # (a) word_given: sample z from full word, (b) subword_given: sample z from subword
    #args.sample_type = 'subword_given'
    # logging
    args.logdir = 'evaluation/morph_segmentation/results/vae/'+args.model_id+'/'+args.recon_type+'/nsamples'+str(args.nsamples)+'/'+args.sample_type+'/'+args.heur_type+'/eps'+str(args.eps)+'/'
    args.fseg   = args.logdir +'segments.txt'
    args.fprob  = args.logdir +'probs.json'
    args.load_probs_from_file = False; args.save_probs_to_file = not args.load_probs_from_file
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")
    # initialize model
    # load vocab (to initialize the model with correct vocabsize)
    with open(model_vocab) as f:
        word2id = json.load(f)
        args.vocab = VocabEntry(word2id)
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.ni = 256; args.nz = 32; 
    args.enc_nh = 256; args.dec_nh = 256
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.model = VAE(args, args.vocab, model_init, emb_init)
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.to(args.device)
    args.model.eval()
    # data
    args.tstdata = 'evaluation/morph_segmentation/data/goldstdsample.tur'
    #args.tstdata = 'evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur'
    args.maxtstsize = 100000
    args.batch_size = 1
    return args


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_id',  required=True)
parser.add_argument('--recon_type', required=True)
parser.add_argument('--sample_type', required=True)      
args = parser.parse_args()

args = config(args)
data, batches = build_data(args)
word_probs = dict()
fseg = open(args.fseg, 'w')
# loop through each word 
for data in batches:
    word = ''.join(args.vocab.decode_sentence(data[0][1:-1]))
    #print(word)
    logps = get_logps(args, word, data, from_file=args.load_probs_from_file)
    word_probs[word] = logps
    # call segmentation heuristic 
    if args.heur_type == 'prev_mid_next_and_prevnext_exceed':
        morphemes = heur_prev_mid_next_and_prevnext_exceed(logps, args.eps)
    elif args.heur_type == 'prev_mid_next':
        morphemes = heur_prev_mid_next(logps, args.eps)
    # write morphemes to file
    fseg.write(str(' '.join(morphemes)+'\n'))     
if args.save_probs_to_file:
    with open(args.fprob, 'w') as json_file:
        json_object = json.dumps(word_probs, indent = 4)
        json_file.write(json_object)
