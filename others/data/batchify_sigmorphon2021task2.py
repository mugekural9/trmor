## Also used for morpho2005

import re, torch, json, os
from data.vocab import  VocabEntry, MonoTextData
from collections import defaultdict

number_of_surf_tokens = 0; number_of_surf_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict(); surfacemap = dict()

def read_valdata(maxdsize, file, surface_vocab, mode):
    surf_data = []; data = []
    all_surfs = dict()
    count = 0
    with open(file, 'r') as reader:
        for line in reader: 
            count += 1
            if count > maxdsize:
                break
            #if line == '\n':
            #    continue
            surfs = line.strip().split(' ')
            #breakpoint()
            for surf in surfs:
                if surf not in all_surfs.keys():
                    all_surfs[surf] = 'added'
                    surf_data.append([surface_vocab[char] for char in surf])
            
    print(mode,':')
    print('surf_data:',  len(surf_data))
    for surf in surf_data:
        data.append([surf])
    return data
    
def get_batch(x, surface_vocab):
    global number_of_surf_tokens, number_of_surf_unks
    surf = []
    max_surf_len = max([len(s[0]) for s in x])
    for surf_idx in x:
        surf_idx = surf_idx[0]
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        # Count statistics...
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad=''):
    continuity = (seq_to_no_pad == '')
    print('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
    # reset dataset statistics
    global number_of_surf_tokens, number_of_surf_unks
    number_of_surf_tokens = 0
    number_of_surf_unks = 0
    order = range(len(data))
    z = zip(order,data)
    if not continuity:
        # 0:sort according to surfaceform, 1: featureform, 3: rootform 
        if seq_to_no_pad == 'surface':
            z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
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
            batches.append(get_batch(data[i: jr], vocab))
            i = jr
        else:
            batches.append(get_batch(data[i: i+batchsize], vocab))
            i += batchsize
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    return batches, order    

def build_data(args):
    # Read data and get batches...
    surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab 
    trndata = read_valdata(args.maxtrnsize, args.trndata, surface_vocab, 'TRN')
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_valdata(args.maxvalsize, args.valdata, surface_vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_valdata(args.maxtstsize, args.tstdata, surface_vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, surface_vocab, args.batchsize, '') 
    #tstdata = None; tst_batches = None
    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), surface_vocab

def log_data(data, dset, surface_vocab, logger, modelname, dsettype='trn'):
    f = open(modelname+'/'+dset+".txt", "w")
    total_surf_len = 0
    for (surf, feat) in data:
        total_surf_len += len(surf)
        f.write(''.join(surface_vocab.decode_sentence_2(surf))+'\n')
    f.close()
    avg_surf_len = total_surf_len / len(data)
    logger.write('%s -- size:%.1d,  avg_surf_len: %.2f \n' % (dset, len(data), avg_surf_len))

