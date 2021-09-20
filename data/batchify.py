import re
import torch
import json
from data import  VocabEntry
from collections import defaultdict

number_of_feat_tokens = 0
number_of_feat_unks = 0
number_of_surf_tokens = 0
number_of_surf_unks = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict()
surfacemap = dict()


def read_trndata_makevocab(file, trnsize, surface_vocab):
    surf_data = []
    feat_data = []

    surf_data_str = []
    feat_data_str = []
    data = []
    with open(file, 'r') as reader:
        feature_vocab = defaultdict(lambda: len(feature_vocab))
        feature_vocab['<pad>'] = 0
        feature_vocab['<s>'] = 1
        feature_vocab['</s>'] = 2
        feature_vocab['<unk>'] = 3

        count = 0
        for line in reader:  
            count += 1
            if count > trnsize:
                break
            split_line = line.split()
            surf_data_str.append(split_line[0]) # add verbal versions
            feat_data_str.append(split_line[1]) # add verbal versions
            surf_data.append([surface_vocab[char] for char in split_line[0]])
            root_and_tags = [char for char in split_line[1].replace('^','+').split('+')[0]] + split_line[1].replace('^','+').split('+')[1:]
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])
        
    with open('trmor_data/feat_vocab.json', 'w') as file:
        file.write(json.dumps(feature_vocab)) 

    with open('trmor_data/surf_vocab.json', 'w') as file:
        file.write(json.dumps(surface_vocab.word2id))
    
    for (surf,feat) in zip(surf_data, feat_data):
        data.append([surf, feat])

    return data, VocabEntry(feature_vocab), (surf_data_str, feat_data_str)
    

def read_valdata(file, feature_vocab, surface_vocab):
    surf_data = []
    feat_data = []
    data = []
    with open(file, 'r') as reader:
        for line in reader:   
            split_line = line.split()
            surf_data.append([surface_vocab[char] for char in split_line[0]])
            root_and_tags = [char for char in split_line[1].replace('^','+').split('+')[0]] + split_line[1].replace('^','+').split('+')[1:]
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])
        
    for (surf,feat) in zip(surf_data, feat_data):
        data.append([surf, feat])

    return data
    
    
def get_batch(x, surface_vocab, feature_vocab):
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks

    surf, feat = [], []

    max_surf_len = max([len(s[0]) for s in x])
    max_feat_len = max([len(s[1]) for s in x])
    
    for surf_idx, feat_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)

        feat_padding = [feature_vocab['<pad>']] * (max_feat_len - len(feat_idx)) 
        feat.append([feature_vocab['<s>']] + feat_idx + [feature_vocab['</s>']] + feat_padding)
        
        # Count statistics...
        number_of_feat_tokens += len(feat_idx)
        number_of_feat_unks += feat_idx.count(feature_vocab['<unk>'])
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
        
    return torch.LongTensor(surf).t().contiguous().to(device), \
        torch.LongTensor(feat).t().contiguous().to(device)  # time * batch

def get_batches(data, surface_vocab, feature_vocab, batchsize=64, seq_to_no_pad='surface'):
    #print('seq not to pad...', seq_to_no_pad)
    # reset dataset statistics
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks
    number_of_feat_tokens = 0
    number_of_feat_unks = 0
    number_of_surf_tokens = 0
    number_of_surf_unks = 0

    order = range(len(data))
    z = zip(order,data)
    '''
    # 0:sort according to surfaceform, 1: featureform 
    if seq_to_no_pad == 'surface':
        z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
    elif seq_to_no_pad == 'feature':
        z = sorted(zip(order, data), key=lambda i: len(i[1][1]))
    '''
    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
        '''
        jr = i
        # data (surfaceform, featureform)
        if seq_to_no_pad == 'surface':
            while jr < min(len(data), i+batchsize) and len(data[jr][0]) == len(data[i][0]): # Do not pad and select equal length of, 0: +surfaceform, 1: +featureform
                jr += 1
        elif seq_to_no_pad == 'feature':
            while jr < min(len(data), i+batchsize) and len(data[jr][1]) == len(data[i][1]): # Do not pad and select equal length of, 0: +surfaceform, 1: +featureform
                jr += 1
        batches.append(get_batch(data[i: jr], surface_vocab, feature_vocab))
        i = jr
        '''
        #print(i, i+batchsize)
        batch = get_batch(data[i: i+batchsize], surface_vocab, feature_vocab)
        
        batches.append(batch)
        i += batchsize

    print('# of feat tokens: ', number_of_feat_tokens, ', # of feat unks: ', number_of_feat_unks)
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)

    return batches, order    
