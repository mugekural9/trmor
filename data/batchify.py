import re
import torch
import json
from data.vocab import  VocabEntry, MonoTextData
from collections import defaultdict

unique_roots = True
unique_surfs = True

number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_feat_tokens = 0; number_of_feat_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict(); surfacemap = dict()

def read_trndata_makevocab(file, trnsize, surface_vocab):
    surf_data = []; feat_data = []; pos_data = []; root_data = [];  data = []
    surf_data_str = []; feat_data_str = []; root_data_str = []
   
    with open(file, 'r') as reader:
        feature_vocab = defaultdict(lambda: len(feature_vocab))
        feature_vocab['<pad>'] = 0
        feature_vocab['<s>'] = 1
        feature_vocab['</s>'] = 2
        feature_vocab['<unk>'] = 3
        pos_vocab = defaultdict(lambda: len(pos_vocab))
        pos_vocab['<unk>'] = 0
        count = 0
        for line in reader:  
            count += 1
            if count > trnsize:
                break
            split_line = line.split()

            root_str = split_line[1].replace('^','+').split('+')[0]
            pos_tag  = split_line[1].replace('^','+').split('+')[1]
            tags     = split_line[1].replace('^','+').split('+')[1:]

            if pos_tag != 'Verb' and pos_tag != 'Noun' and pos_tag != 'Adj':
                continue
            
            surf_data_str.append(split_line[0]) # add verbal versions
            feat_data_str.append(split_line[1]) # add verbal versions
            root_data_str.append(root_str)      # add verbal versions

            # fill surface vocab
            surf_data.append([surface_vocab[char] for char in split_line[0]])
            root = [char for char in root_str]
            root_and_tags =  root + tags
            root_data.append([surface_vocab[char] for char in root])
           
            # fill feature vocab
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])

            # fill pos vocab
            pos_data.append(pos_vocab[pos_tag])

    with open('trmor_data/surf_vocab.json', 'w') as file:
        file.write(json.dumps(surface_vocab.word2id)) 
    
    with open('trmor_data/feat_vocab.json', 'w') as file:
        file.write(json.dumps(feature_vocab)) 
    
    with open('trmor_data/pos_vocab.json', 'w') as file:
        file.write(json.dumps(pos_vocab))


    added_roots = []
    added_surfs = []
    for (surf, feat, pos, root, root_str) in zip(surf_data, feat_data, pos_data, root_data, root_data_str):
        if unique_surfs and surf in added_surfs:
            continue
        else:
            added_surfs.append(surf)
            data.append([surf, feat, [pos], root])

    return data, VocabEntry(feature_vocab), VocabEntry(pos_vocab), (surf_data_str, feat_data_str)
    
def read_valdata(file, surface_vocab, feature_vocab, pos_vocab):
    surf_data = []; feat_data = []; pos_data = []
    root_data = []; root_data_str = []; data = []
    with open(file, 'r') as reader:
        for line in reader:   
            split_line = line.split()
            surf_data.append([surface_vocab[char] for char in split_line[0]])
            root_data_str.append(split_line[1].replace('^','+').split('+')[0])
            root = [char for char in split_line[1].replace('^','+').split('+')[0]]
            root_and_tags =  root + split_line[1].replace('^','+').split('+')[1:]
            root_data.append([surface_vocab[char] for char in root])
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])
            pos_tag = split_line[1].replace('^','+').split('+')[1]
            pos_data.append(pos_vocab[pos_tag])

    added_roots = []
    for (surf, feat, pos, root, root_str) in zip(surf_data, feat_data, pos_data, root_data, root_data_str):
        if unique_roots and root_str in added_roots:
            continue
        else:
            added_roots.append(root_str)
            data.append([surf, feat, [pos], root])
    return data
    
def get_batch(x, surface_vocab, feature_vocab, pos_vocab):
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks

    surf, feat, pos, root = [], [], [], []

    max_surf_len = max([len(s[0]) for s in x])
    max_feat_len = max([len(s[1]) for s in x])
    max_root_len = max([len(s[3]) for s in x])

    for surf_idx, feat_idx, pos_idx, root_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)

        feat_padding = [feature_vocab['<pad>']] * (max_feat_len - len(feat_idx)) 
        feat.append([feature_vocab['<s>']] + feat_idx + [feature_vocab['</s>']] + feat_padding)
        
        #pos.append([pos_vocab['<s>']] + pos_idx + [pos_vocab['</s>']])
        pos.append(pos_idx)
        
        root_padding = [surface_vocab['<pad>']] * (max_root_len - len(root_idx)) 
        root.append([surface_vocab['<s>']] + root_idx + [surface_vocab['</s>']] + root_padding)

        # Count statistics...
        number_of_feat_tokens += len(feat_idx)
        number_of_feat_unks += feat_idx.count(feature_vocab['<unk>'])
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])

    return  torch.tensor(surf, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(feat, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(pos,  dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(root, dtype=torch.long, requires_grad=False, device=device)

def get_batches(data, surface_vocab, feature_vocab, pos_vocab, batchsize=64, seq_to_no_pad=''):

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
            batches.append(get_batch(data[i: jr], surface_vocab, feature_vocab, pos_vocab))
            i = jr
        else:
            batch = get_batch(data[i: i+batchsize], surface_vocab, feature_vocab, pos_vocab)
            batches.append(batch)
            i += batchsize

    print('# of feat tokens: ', number_of_feat_tokens, ', # of feat unks: ', number_of_feat_unks)
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)

    return batches, order    

def build_data(args):
    # Read data and get batches...
    surface_vocab = MonoTextData(args.trndata, label=False).vocab
    _trndata, feature_vocab, pos_vocab, _ = read_trndata_makevocab(args.trndata, args.trnsize, surface_vocab) # data len:56729 unique surfaces
    #trndata = _trndata[2000:10000]  
    #vlddata = _trndata[:2000] 
    
    trndata = _trndata[3829:] # 52000 instances
    vlddata = _trndata[:3829] # 3829  instances
    
    tstdata = None
    #vlddata = read_valdata(args.valdata, surface_vocab, feature_vocab, pos_vocab)     # 5000
    #tstdata = read_valdata(args.tstdata, surface_vocab, feature_vocab, pos_vocab)     # 2769

    trn_batches, _ = get_batches(trndata, surface_vocab, feature_vocab, pos_vocab, args.batchsize, args.seq_to_no_pad) 
    vld_batches, _ = get_batches(vlddata, surface_vocab, feature_vocab, pos_vocab, args.batchsize, args.seq_to_no_pad) 
    #tst_batches, _ = get_batches(tstdata, surface_vocab, feature_vocab, pos_vocab, args.batchsize, args.seq_to_no_pad)) 
    tst_batches = None

    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), (surface_vocab, feature_vocab, pos_vocab)


def log_data(data, dset, surface_vocab, feature_vocab, pos_vocab, logger):
    # logger.write out data to file
    f = open(dset+".txt", "w")
    total_surf_len = 0
    total_feat_len = 0
    total_root_len = 0
    pos_tags = defaultdict(lambda: 0)
    for (surf, feat, pos, root) in data:
        total_surf_len += len(surf)
        total_feat_len += len(feat)
        total_root_len += len(root)
        pos_tags[pos[0]] += 1
        f.write(''.join(surface_vocab.decode_sentence_2(surf))+'\t'+''.join(pos_vocab.decode_sentence_2(pos))+'\n')
    f.close()
    avg_surf_len = total_surf_len / len(data)
    avg_feat_len = total_feat_len / len(data)
    avg_root_len = total_root_len / len(data)
    #print('%s -- size:%.1d, avg_surf_len: %.2f,  avg_feat_len: %.2f, avg_root_len: %.2f \n' % (dset, len(data), avg_surf_len, avg_feat_len, avg_root_len))
    logger.write('%s -- size:%.1d, avg_surf_len: %.2f,  avg_feat_len: %.2f, avg_root_len: %.2f \n' % (dset, len(data), avg_surf_len, avg_feat_len, avg_root_len))
    for tag,count in sorted(pos_tags.items(), key=lambda item: item[1], reverse=True):
        logger.write('Pos tag %s: %.4f \n' % (pos_vocab.id2word(tag), count/len(data)))
        #print('Pos tag %s: %.4f \n' % (pos_vocab.id2word(tag), count/sum(pos_tags.values())))

