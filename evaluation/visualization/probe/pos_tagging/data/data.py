import re, torch, json, os
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches

def read_data(maxdsize, file, vocab, mode):
    surface_vocab,  surfpos_vocab = vocab
    surf_data = []; surfpos_data = []; data = []
    all_surfs = dict()
    count = 0
    if 'surf.uniquesurfs' in file:
        with open(file, 'r') as reader:
            for line in reader: 
                count += 1
                if count > maxdsize:
                    break
                surf, tags = line.strip().split('\t')
                tags     = tags.replace('^','+').split('+')[1:]
                surf_data.append([surface_vocab[char] for char in surf])

                # fill surfpos vocab and data
                surfpostags = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Ques', 'Num', 'Postp', 'Conj', 'Dup', 'Interj', 'Det'] 
                surfpos_info_exists = False
                for tag in reversed(tags):
                    if tag in surfpostags:
                        surfpos_data.append(surfpos_vocab[tag])
                        surfpos_info_exists = True
                        break
                if not surfpos_info_exists:
                    surfpos_data.append(surfpos_vocab['<unk>'])
       
    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('surfpos_data:', len(surfpos_data))
    assert len(surf_data) == len(surfpos_data)
    for (surf, surfpos) in zip(surf_data,  surfpos_data):
        data.append([surf, [surfpos]])
    return data

def get_batch(x, vocab, device='cpu'):
    global number_of_surf_tokens, number_of_surfpos_tokens, number_of_surf_unks, number_of_surfpos_unks
    surface_vocab,  surfpos_vocab = vocab
    surf, surfpos = [], []
    max_surf_len = max([len(s[0]) for s in x])
    for surf_idx, surfpos_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        surfpos.append(surfpos_idx)
        # Count statistics...
        number_of_surf_tokens += len(surf_idx);  number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
        number_of_surfpos_tokens += len(surfpos_idx);  number_of_surfpos_unks += surfpos_idx.count(surfpos_vocab['<unk>'])
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(surfpos, dtype=torch.long, requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad='', device='cpu'):
    continuity = (seq_to_no_pad == '')
    print('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
    # reset dataset statistics
    global  number_of_surf_tokens, number_of_surfpos_tokens, number_of_surf_unks, number_of_surfpos_unks
    number_of_surf_tokens = 0
    number_of_surf_unks = 0
    number_of_surfpos_tokens = 0
    number_of_surfpos_unks = 0
    order = range(len(data))
    z = zip(order,data)
    if not continuity:
        # 0:sort according to surfaceform, 1: featureform, 3: rootform 
        if seq_to_no_pad == 'surface':
            z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
        elif seq_to_no_pad == 'feature':
            z = sorted(zip(order, data), key=lambda i: len(i[1][1]))
        elif seq_to_no_pad == 'root':
            z = sorted(zip(order, data), key=lambda i: len(i[1][3]))
    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
        if not continuity:
            jr = i
            # data (surfaceform, featureform)
            if seq_to_no_pad == 'surface':
                while jr < min(len(data), i+batchsize) and len(data[jr][0]) == len(data[i][0]): # Do not pad and select equal length of, 0: +surfaceform, 1: +featureform, 2:rootpostag, 3: +root
                    jr += 1
            elif seq_to_no_pad == 'feature':
                while jr < min(len(data), i+batchsize) and len(data[jr][1]) == len(data[i][1]): 
                    jr += 1
            elif seq_to_no_pad == 'root':
                while jr < min(len(data), i+batchsize) and len(data[jr][3]) == len(data[i][3]): 
                    jr += 1
            batches.append(get_batch(data[i: jr], vocab, device=device))
            i = jr
        else:
            batches.append(get_batch(data[i: i+batchsize], vocab, device=device))
            i += batchsize
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    print('# of surfpos tokens: ', number_of_surfpos_tokens, ', # of surfpos unks: ', number_of_surfpos_unks)
    
    return batches, order    

def build_data(args):
   # Read data and get batches...
    tst_data = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')
    tst_batches, _ = get_batches(tst_data, args.vocab, args.batch_size, '', args.device) 
    return tst_data, tst_batches