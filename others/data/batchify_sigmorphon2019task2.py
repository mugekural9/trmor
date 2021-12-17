import re, torch, json, os
from data.vocab import  VocabEntry, MonoTextData
from collections import defaultdict

number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_feat_tokens = 0; number_of_feat_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict(); surfacemap = dict()

def read_trndata_makevocab(args, surface_vocab):
    surf_data = []; lemma_plus_feat_data = [];  data = []
    with open(args.trndata, 'r') as reader:
        feature_vocab = defaultdict(lambda: len(feature_vocab))
        feature_vocab['<pad>'] = 0
        feature_vocab['<s>'] = 1
        feature_vocab['</s>'] = 2
        feature_vocab['<unk>'] = 3
        count = 0
        surf_d = None; lemma_plus_feat_d = None
        for line in reader: 
            if line == '\n':
                continue
            if '#' in line:
                if surf_d == None and lemma_plus_feat_d == None:
                    surf_d = []; lemma_plus_feat_d = []
                    continue
                flat_surf = [item for sublist in surf_d for item in sublist]
                flat_lemma_plus_feat_d = [item for sublist in lemma_plus_feat_d for item in sublist]
                # fill surface data
                surf_data.append(flat_surf)
                # fill feature data
                lemma_plus_feat_data.append(flat_lemma_plus_feat_d)
                surf_d = []; lemma_plus_feat_d = []
                count += 1
                continue
            split_line = line.strip().split('\t')
            surf = split_line[1]
            lemma = split_line[2]
            feat = split_line[5]
            surf_d.append([surface_vocab[char] for char in surf])
            lemma_plus_feat_d.append([feature_vocab[char] for char in lemma] + [feature_vocab[feat]])

            if count > args.maxtrnsize:
                break
         
        
    print('TRN:')
    print('surf_data:',  len(surf_data))
    print('lemma_plus_feat_data:',  len(lemma_plus_feat_data))
    assert len(surf_data) == len(lemma_plus_feat_data) 
    for (surf, lemma_plus_feat) in zip(surf_data, lemma_plus_feat_data):
        data.append([surf, lemma_plus_feat])
    vocab = (surface_vocab, VocabEntry(feature_vocab))

    # Logging
    args.modelname = 'logs/'+args.task+'/'+args.bmodel+'/'+str(len(data))+'_instances'
    try:
        os.makedirs(args.modelname)
        print("Directory " , args.modelname ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.modelname ,  " already exists")

    with open(args.modelname+'/surf_vocab.json', 'w') as file:
        file.write(json.dumps(surface_vocab.word2id, ensure_ascii=False)) 
    with open(args.modelname+'/feat_vocab.json', 'w') as file:
        file.write(json.dumps(feature_vocab, ensure_ascii=False)) 
    return  data, vocab, (None, None)
    
def read_valdata(maxdsize, file, vocab, mode):
    surface_vocab, feature_vocab = vocab
    surf_data = []; lemma_plus_feat_data = [];  data = []
    count = 0
    surf_d = None; lemma_plus_feat_d = None
    with open(file, 'r') as reader:
        for line in reader: 
            if line == '\n':
                continue
            if '#' in line:
                if surf_d == None and lemma_plus_feat_d == None:
                    surf_d = []; lemma_plus_feat_d = []
                    continue
                flat_surf = [item for sublist in surf_d for item in sublist]
                flat_lemma_plus_feat_d = [item for sublist in lemma_plus_feat_d for item in sublist]
                # fill surface data
                surf_data.append(flat_surf)
                # fill feature data
                lemma_plus_feat_data.append(flat_lemma_plus_feat_d)
                surf_d = []; lemma_plus_feat_d = []
                count += 1
                continue
            split_line = line.strip().split('\t')
            surf = split_line[1]
            lemma = split_line[2]
            feat = split_line[5]
            surf_d.append([surface_vocab[char] for char in surf])
            lemma_plus_feat_d.append([feature_vocab[char] for char in lemma] + [feature_vocab[feat]])
            if count > maxdsize:
                break

    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('lemma_plus_feat_data:',  len(lemma_plus_feat_data))
    assert len(surf_data) == len(lemma_plus_feat_data)
    for (surf, lemma_plus_feat) in zip(surf_data, lemma_plus_feat_data):
        data.append([surf, lemma_plus_feat])
    return data
    
def get_batch(x, vocab):
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks

    surface_vocab, feature_vocab = vocab

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

    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(feat, dtype=torch.long,  requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad=''):
    continuity = (seq_to_no_pad == '')
    print('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
    # reset dataset statistics
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks
    number_of_feat_tokens = 0
    number_of_feat_unks = 0
    number_of_surf_tokens = 0
    number_of_surf_unks = 0
    order = range(len(data))
    z = zip(order,data)
    if not continuity:
        # 0:sort according to surfaceform, 1: featureform, 3: rootform 
        if seq_to_no_pad == 'surface':
            z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
        elif seq_to_no_pad == 'feature':
            z = sorted(zip(order, data), key=lambda i: len(i[1][1]))
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
    print('# of feat tokens: ', number_of_feat_tokens, ', # of feat unks: ', number_of_feat_unks)
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    return batches, order    

def build_data(args):
    # Read data and get batches...
    surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab 
    trndata, vocab, _ = read_trndata_makevocab(args, surface_vocab) 
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_valdata(args.maxvalsize, args.valdata, vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_valdata(args.maxtstsize, args.tstdata, vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, vocab, args.batchsize, '') 
    #tstdata = None; tst_batches = None

    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), vocab

def log_data(data, dset, vocab, logger, modelname, dsettype='trn'):
    surface_vocab, feature_vocab = vocab
    f = open(modelname+'/'+dset+".txt", "w")
    total_surf_len = 0; total_feat_len = 0
    for (surf, feat) in data:
        total_surf_len += len(surf)
        total_feat_len += len(feat)
        f.write(
        ''.join(surface_vocab.decode_sentence_2(surf))+'\t'
        +''.join(feature_vocab.decode_sentence_2(feat))+'\n')
    f.close()
    avg_surf_len = total_surf_len / len(data)
    avg_feat_len = total_feat_len / len(data)
    logger.write('%s -- size:%.1d,  avg_surf_len: %.2f, avg_feat_len: %.2f  \n' % (dset, len(data), avg_surf_len, avg_feat_len))

