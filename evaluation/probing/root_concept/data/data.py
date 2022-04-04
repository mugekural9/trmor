import re, torch, json, os
from common.vocab import  VocabEntry
from collections import defaultdict

number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_root_concept_tokens = 0; number_of_root_concept_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def read_trndata_makevocab(args, surf_vocab):
    data = []; surf_data = []; root_concept_data = []
   
    with open(args.trndata, 'r') as reader:
        root_concept_vocab =  defaultdict(lambda: len(root_concept_vocab))
        root_concept_vocab['<unk>'] = 0
        count = 0
        for line in reader: 
            count += 1
            if count > args.maxtrnsize:
                break
            split_line = line.split()
            tag     = split_line[1].split('+')[0]
            root_concept_data.append(root_concept_vocab[tag])
            # fill surface vocab and data
            surf_data.append([surf_vocab[char] for char in split_line[0]])
    print('TRN:')
    print('surf_data:',  len(surf_data))
    print('root_concept_data:', len(root_concept_data))

    assert len(surf_data) == len(root_concept_data)
    for (surf, root_concept) in zip(surf_data,  root_concept_data):
        data.append([surf, [root_concept]])
    return  data, VocabEntry(root_concept_vocab)
    
def read_valdata(maxdsize, file, vocab, mode):
    surf_vocab, root_concept_vocab = vocab
    data = []; surf_data = []; root_concept_data = []
    count = 0
    with open(file, 'r') as reader:
        for line in reader:   
            count += 1
            if count > maxdsize:
                break
            split_line = line.split()
            surf_data.append([surf_vocab[char] for char in split_line[0]])
            tag     = split_line[1].split('+')[0]
            root_concept_data.append(root_concept_vocab[tag])

    mydict = dict()
    for i in root_concept_data:
        mydict[i]=1
    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('root_concept_data:', len(root_concept_data))
    assert len(surf_data) == len(root_concept_data) 
    for (surf, root_concept) in zip(surf_data, root_concept_data):
        data.append([surf, [root_concept]])
    return data
    
def get_batch(x, vocab):
    global number_of_surf_tokens, number_of_root_concept_tokens, number_of_surf_unks, number_of_root_concept_unks
    surface_vocab,  root_concept_vocab = vocab
    surf, root_concept = [], []
    max_surf_len = max([len(s[0]) for s in x])
    for surf_idx, root_concept_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        root_concept.append(root_concept_idx)
        # Count statistics...
        number_of_surf_tokens += len(surf_idx);  number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
        number_of_root_concept_tokens += len(root_concept_idx);  number_of_root_concept_unks += root_concept_idx.count(root_concept_vocab['<unk>'])
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(root_concept, dtype=torch.long, requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad=''):
    continuity = (seq_to_no_pad == '')
    #print('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
    # reset dataset statistics
    global  number_of_surf_tokens, number_of_root_concept_tokens, number_of_surf_unks, number_of_root_concept_unks
    number_of_surf_tokens = 0
    number_of_surf_unks = 0
    number_of_root_concept_tokens = 0
    number_of_root_concept_unks = 0
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
            batches.append(get_batch(data[i: jr], vocab))
            i = jr
        else:
            batches.append(get_batch(data[i: i+batchsize], vocab))
            i += batchsize
    #print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    #print('# of root_concept tokens: ', number_of_root_concept_tokens, ', # of root_concept unks: ', number_of_root_concept_unks)
    
    return batches, order    

def build_data(args, surf_vocab):
    # Read data and get batches...
    trndata, root_concept_vocab = read_trndata_makevocab(args, surf_vocab) 
    vocab = (surf_vocab, root_concept_vocab)
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_valdata(args.maxvalsize, args.valdata, vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_valdata(args.maxtstsize, args.tstdata, vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, vocab, args.batchsize, args.seq_to_no_pad) 
    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), vocab
