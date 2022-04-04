import re, torch, json, os
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches

def read_data(maxdsize, file, surface_vocab, mode):
    surf_data = []; polar_data = []; data = []
    all_surfs = dict()
    count = 0
    if 'unique' in file:
        with open(file, 'r') as reader:
            for line in reader: 
                count += 1
                if count > maxdsize:
                    break
                surf, tags = line.strip().split('\t')
                tags     = tags.replace('^','+').split('+')[1:]
                surf_data.append([surface_vocab[char] for char in surf])
    elif 'wordlist.tur' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split(' ')[1]
                surf_data.append([surface_vocab[char] for char in surf])
    print(mode,':')
    print('surf_data:',  len(surf_data))
    for surf in surf_data:
        data.append([surf])
    return data
    
def build_data(args):
   # Read data and get batches...
    tst_data = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')
    tst_batches, _ = get_batches(tst_data, args.vocab, args.batch_size, '', args.device) 
    return tst_data, tst_batches