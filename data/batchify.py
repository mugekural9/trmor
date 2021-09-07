import re
import torch
import json
from data import Vocab

number_of_feat_tokens = 0
number_of_feat_unks = 0
number_of_surf_tokens = 0
number_of_surf_unks = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict()
surfacemap = dict()


def readdata(trnsize):
    snts = dict()
    data = []
    with open('trmor_data/trmor2018.train', 'r') as reader:
       
        # Read and print the entire file line by line
        snt = []
        sent_state = False
        for line in reader:   
            if '<S id' in line:
                s_id = re.search('id="(.*)"', line).group(1) 
                sent_state = True
                snt = []
                continue
            elif "</S>" in line:
                snts[s_id] = snt
                data.extend(snt)
                sent_state = False
                continue
            if sent_state:
                splt_line = line.lower().split()
                if len(splt_line) > 2: # multiple data 
                    for i in range(1,len(splt_line)):
                        if splt_line[i] not in datamap and splt_line[0] not in surfacemap:
                            surfacemap[splt_line[0]] = 'here'
                            datamap[splt_line[i]] = splt_line[0]
                            tags = splt_line[i].replace('^','+').split('+') 
                            tags = [char for char in tags[0]] + tags[1:]
                            snt.append([splt_line[0], tags])
                else:
                    if splt_line[1] not in datamap  and splt_line[0] not in surfacemap:
                        surfacemap[splt_line[0]] = 'here'
                        datamap[splt_line[1]] = splt_line[0]
                        tags = splt_line[1].replace('^','+').split('+')
                        tags = [char for char in tags[0]] + tags[1:]
                        snt.append([splt_line[0], tags])

    vocab_file = 'trmor_data/feature_vocab.txt'    
    Vocab.build(data[:trnsize], vocab_file, 10000)
    vocab = Vocab(vocab_file)
    with open('trmor_data/feature_vocab.json', 'w') as file:
        file.write(json.dumps(vocab.word2id)) 
    # data: [[surfaceform, featureform]]
    return data, vocab
    
def get_batch(x, surface_vocab, feature_vocab):
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks

    surf, feat = [], []

    max_surf_len = max([len(s[0]) for s in x])
    max_feat_len = max([len(s[1]) for s in x])
    
    for snt in x:
        surf_str = snt[0]
        feat_str  = snt[1]
        surf_str = [char for char in surf_str]        
        
        surf_idx =  [surface_vocab.word2id[w] if w in surface_vocab.word2id else surface_vocab['<unk>'] for w in surf_str]
        feat_idx =  [feature_vocab.word2id[w] if w in feature_vocab.word2id else feature_vocab.unk for w in feat_str]  
        
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_str)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)

        feat_padding = [feature_vocab.pad] * (max_feat_len - len(feat_str)) 
        feat.append([feature_vocab.go] + feat_idx + [feature_vocab.eos] + feat_padding)
        
        # Count statistics...
        number_of_feat_tokens += len(feat_idx)
        number_of_feat_unks += feat_idx.count(feature_vocab.unk)
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
        
    return torch.LongTensor(surf).t().contiguous().to(device), \
        torch.LongTensor(feat).t().contiguous().to(device)  # time * batch

def get_batches(data, surface_vocab, feature_vocab, batchsize=64, seq_to_no_pad='feature'):
    order = range(len(data))

    # 0:sort according to surfaceform, 1: featureform 
    if seq_to_no_pad == 'surface':
        z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
    elif seq_to_no_pad == 'feature':
        z = sorted(zip(order, data), key=lambda i: len(i[1][1]))

    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
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

    print('# of feat tokens: ', number_of_feat_tokens, ', # of feat unks: ', number_of_feat_unks)
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)

    return batches, order    
