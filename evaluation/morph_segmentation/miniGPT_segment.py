# -----------------------------------------------------------
# Date:        2022/02/01 
# Author:      Muge Kural
# Description: Morpheme segmentation heuristics for trained charlm model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from models.gpt3 import GPT3
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
            logps[word] = np.exp(args.model.log_probability(data).item())
            # loop through word's subwords 
            for i in range(len(data[0])-2, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                logps[subword] = np.exp(args.model.log_probability(subdata.unsqueeze(0)).item())
        logps = dict(reversed(list(logps.items())))
        return logps


def config():
     # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'miniGPT'
    model_path, model_vocab  = get_model_info(model_id)
    # heuristic
    args.heur_type = 'prev_mid_next'; args.eps = 0.0
    # (a) avg: averages ll over word tokens, (b) sum: adds ll over word tokens
    args.recon_type = 'avg' 
    # logging
    args.logdir = '/kuacc/users/eacikgoz17/NLP/trmor/evaluation/morph_segmentation/results/charlm/'+model_id+'/'+args.recon_type+'/'+args.heur_type+'/eps'+str(args.eps)+'/'
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
    

    num_layers=3
    embed_dim=128
    num_heads=16
    block_size=128
    embedding_dropout_rate=0.15 
    attention_dropout_rate=0.15
    residual_dropout_rate=0.15
    expand_ratio = 4
    args.model = GPT3(vocab=args.vocab,
                      num_layers=num_layers,
                      embed_dim=embed_dim,
                      num_heads=num_heads,
                      block_size=block_size,
                      embedding_dropout_rate=embedding_dropout_rate,
                      attention_dropout_rate=attention_dropout_rate,
                      residual_dropout_rate=residual_dropout_rate,
                      expand_ratio=expand_ratio
                     )

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.to(args.device)
    args.model.eval()
    # data
    args.tstdata = '/kuacc/users/eacikgoz17/NLP/trmor/evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur'
    args.maxtstsize = 40000
    args.batch_size = 1
    return args

def main():
    args = config()
    data, batches = build_data(args)
    word_probs = dict()
    fseg = open(args.fseg, 'w')
    # loop through each word 
    for data in batches:
        word = ''.join(args.vocab.decode_sentence(data[0][1:-1]))
        print(word)
        logps = get_logps(args, word, data, from_file=args.load_probs_from_file)
        word_probs[word] = logps
        # call segmentation heuristic 
        if args.heur_type == 'prev_mid_next':
            morphemes = heur_prev_mid_next(logps, args.eps)
        elif args.heur_type == 'prev_mid_next_and_prevnext_exceed':
            morphemes = heur_prev_mid_next_and_prevnext_exceed(logps, args.eps)
        # write morphemes to file
        fseg.write(str(' '.join(morphemes)+'\n'))     
    if args.save_probs_to_file:
        with open(args.fprob, 'w') as json_file:
            json_object = json.dumps(word_probs, indent = 4)
            json_file.write(json_object)

if __name__=="__main__":
    main()
