import re, torch, json, os
from common.vocab import  VocabEntry
from collections import defaultdict

number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_surfpos_tokens = 0; number_of_surfpos_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def read_trndata_makevocab(args, surf_vocab):
    data = []; surf_data = []; surfpos_data = []
   
    with open(args.trndata, 'r') as reader:
        surfpos_vocab =  defaultdict(lambda: len(surfpos_vocab))
        surfpos_vocab['<unk>'] = 0
        count = 0
        for line in reader: 
            count += 1
            if count > args.maxtrnsize:
                break
            split_line = line.split()
            tags     = split_line[1].replace('^','+').split('+')[1:]
            # fill surface vocab and data
            surf_data.append([surf_vocab[char] for char in split_line[0]])
            # fill surfpos vocab and data
            surfpostags = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Ques', 'Num', 'Postp', 'Conj', 'Dup', 'Interj', 'Det'] 
            for tag in reversed(tags):
                if tag in surfpostags:
                    surfpos_data.append(surfpos_vocab[tag])
                    break
    print('TRN:')
    print('surf_data:',  len(surf_data))
    print('surfpos_data:', len(surfpos_data))

    assert len(surf_data) == len(surfpos_data)
    for (surf, surfpos) in zip(surf_data,  surfpos_data):
        data.append([surf, [surfpos]])
    return  data, VocabEntry(surfpos_vocab)
    
def read_valdata(maxdsize, file, vocab, mode):
    surf_vocab, surfpos_vocab = vocab
    data = []; surf_data = []; surfpos_data = []
    count = 0
    with open(file, 'r') as reader:
        for line in reader:   
            count += 1
            if count > maxdsize:
                break
            split_line = line.split()
            tags     = split_line[1].replace('^','+').split('+')[1:]
            surf_data.append([surf_vocab[char] for char in split_line[0]])
            # fill surfpos data
            surfpostags = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Ques', 'Num', 'Postp', 'Conj', 'Dup', 'Interj', 'Det'] 
            for tag in reversed(tags):
                if tag in surfpostags:
                    surfpos_data.append(surfpos_vocab[tag])
                    break
    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('surfpos_data:', len(surfpos_data))
    assert len(surf_data) == len(surfpos_data) 
    for (surf, surfpos) in zip(surf_data, surfpos_data):
        data.append([surf, [surfpos]])
    return data
    
def get_batch(x, vocab):
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

def get_batches(data, vocab, batchsize=64, seq_to_no_pad=''):
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
            batches.append(get_batch(data[i: jr], vocab))
            i = jr
        else:
            batches.append(get_batch(data[i: i+batchsize], vocab))
            i += batchsize
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    print('# of surfpos tokens: ', number_of_surfpos_tokens, ', # of surfpos unks: ', number_of_surfpos_unks)
    
    return batches, order    

def build_data(args, surf_vocab):
    # Read data and get batches...
    trndata, surfpos_vocab = read_trndata_makevocab(args, surf_vocab) 
    vocab = (surf_vocab, surfpos_vocab)
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_valdata(args.maxvalsize, args.valdata, vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_valdata(args.maxtstsize, args.tstdata, vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, vocab, args.batchsize, args.seq_to_no_pad) 
    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), vocab

def log_data(data, dset, vocab, logger, modelname, dsettype='trn'):
    surface_vocab, feature_vocab, pos_vocab, polar_vocab, tense_vocab, surfpos_vocab = vocab
    f = open(modelname+'/'+dset+".txt", "w")
    total_surf_len = 0; total_feat_len = 0; total_root_len = 0
    pos_tags, polar_tags, tense_tags, surfpos_tags = defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0), defaultdict(lambda: 0)
    for (surf, feat, pos, root, polar, tense, surfpos) in data:
        total_surf_len += len(surf)
        total_feat_len += len(feat)
        total_root_len += len(root)
        pos_tags[pos[0]]     += 1
        polar_tags[polar[0]] += 1       
        tense_tags[tense[0]] += 1
        surfpos_tags[surfpos[0]] += 1

        f.write(
        ''.join(surface_vocab.decode_sentence_2(surf))+'\t'
        +''.join(surface_vocab.decode_sentence_2(root))+'\t'
        +''.join(pos_vocab.decode_sentence_2(pos))+'\t'
        +''.join(surfpos_vocab.decode_sentence_2(surfpos))+'\n')
        #'feat: ' + ''.join(feature_vocab.decode_sentence_2(feat))+'\t'
        #'polar: ' +''.join(polar_vocab.decode_sentence_2(polar))+'\t'
        #'tense: ' +''.join(tense_vocab.decode_sentence_2(tense))+'\n')
        
    f.close()
    avg_surf_len = total_surf_len / len(data)
    avg_feat_len = total_feat_len / len(data)
    avg_root_len = total_root_len / len(data)
    logger.write('%s -- size:%.1d,  avg_surf_len: %.2f, avg_feat_len: %.2f, avg_root_len: %.2f \n' % (dset, len(data), avg_surf_len, avg_feat_len, avg_root_len))
    
    for tag,count in sorted(pos_tags.items(), key=lambda item: item[1], reverse=True):
        logger.write('Pos tag %s: %.2f, count: %.2d\n' % (pos_vocab.id2word(tag), count/len(data), count))

    for tag,count in sorted(polar_tags.items(), key=lambda item: item[1], reverse=True):
        logger.write('Polarity tag %s: %.2f, count: %.2d \n' % (polar_vocab.id2word(tag), count/len(data), count))

    for tag,count in sorted(tense_tags.items(), key=lambda item: item[1], reverse=True):
        logger.write('Tense tag %s: %.2f, count: %.2d \n' % (tense_vocab.id2word(tag), count/len(data), count))
   
    for tag,count in sorted(surfpos_tags.items(), key=lambda item: item[1], reverse=True):
        logger.write('Surfpos tag %s: %.2f, count: %.2d \n' % (surfpos_vocab.id2word(tag), count/len(data), count))
