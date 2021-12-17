import re, torch, json, os
from data.vocab import  VocabEntry, MonoTextData
from collections import defaultdict

unique_roots = True; unique_surfs = True
number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_feat_tokens = 0; number_of_feat_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict(); surfacemap = dict()

def read_trndata_makevocab(args, surface_vocab):
    surf_data = []; feat_data = []; pos_data = []; root_data = [];  data = []; polar_data = []; tense_data = []; surfpos_data = []
    surf_data_str = []; feat_data_str = []; root_data_str = []
   
    with open(args.trndata, 'r') as reader:
        feature_vocab = defaultdict(lambda: len(feature_vocab))
        feature_vocab['<pad>'] = 0
        feature_vocab['<s>'] = 1
        feature_vocab['</s>'] = 2
        feature_vocab['<unk>'] = 3
        pos_vocab   = defaultdict(lambda: len(pos_vocab))
        polar_vocab = defaultdict(lambda: len(polar_vocab))
        tense_vocab = defaultdict(lambda: len(tense_vocab))
        surfpos_vocab =  defaultdict(lambda: len(surfpos_vocab))
        pos_vocab['<unk>'] = 0; polar_vocab['<unk>'] = 0; tense_vocab['<unk>'] = 0; surfpos_vocab['<unk>'] = 0
        count = 0
        for line in reader: 
            count += 1
            if count > args.maxtrnsize:
                break
            split_line = line.split()
            root_str = split_line[1].replace('^','+').split('+')[0]
            pos_tag  = split_line[1].replace('^','+').split('+')[1]
            tags     = split_line[1].replace('^','+').split('+')[1:]
            surf_data_str.append(split_line[0]) # add verbal versions
            feat_data_str.append(split_line[1]) # add verbal versions
            root_data_str.append(root_str)      # add verbal versions

            # fill surface vocab and data
            surf_data.append([surface_vocab[char] for char in split_line[0]])

            # fill root data
            root = [char for char in root_str]
            root_and_tags =  root + tags
            root_data.append([surface_vocab[char] for char in root])
           
            # fill feature vocab and data
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])

            # fill pos vocab and data
            pos_data.append(pos_vocab[pos_tag])

            # fill polar vocab and data
            polars = ['Neg', 'Pos']
            polar_info_exists = False
            for tag in reversed(tags):
                if tag in polars:
                    polar_data.append(polar_vocab[tag])
                    polar_info_exists = True
                    break
            if not polar_info_exists:
                polar_data.append(polar_vocab['<unk>'])

            # fill tense vocab and data
            tenses = ['Past', 'Narr', 'Fut', 'Prog1', 'Aor']
            tense_info_exists = False
            for tag in reversed(tags):
                if tag in tenses:
                    tense_data.append(tense_vocab[tag])
                    tense_info_exists = True
                    break
            if not tense_info_exists:
                tense_data.append(tense_vocab['<unk>'])

            # fill surfpos vocab and data
            #surfpostags = ['Verb', 'Noun', 'Adj', 'Ques'] 
            surfpostags = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Ques', 'Num', 'Postp', 'Conj', 'Dup', 'Interj', 'Det'] 
            for tag in reversed(tags):
                if tag in surfpostags:
                    surfpos_data.append(surfpos_vocab[tag])
                    break
    print('TRN:')
    print('surf_data:',  len(surf_data))
    print('feat_data:',  len(feat_data))
    print('root_data:',  len(root_data))
    print('pos_data:',   len(pos_data))
    print('polar_data:', len(polar_data))
    print('tense_data:', len(tense_data))
    print('surfpos_data:', len(surfpos_data))
    assert len(surf_data) == len(feat_data) == len(root_data) == len(pos_data) == len(polar_data) == len(tense_data) == len(surfpos_data)
    for (surf, feat, pos, root, polar, tense, surfpos) in zip(surf_data, feat_data, pos_data, root_data, polar_data, tense_data, surfpos_data):
        data.append([surf, feat, [pos], root, [polar], [tense], [surfpos]])
    vocab = (surface_vocab, VocabEntry(feature_vocab), VocabEntry(pos_vocab), VocabEntry(polar_vocab), VocabEntry(tense_vocab), VocabEntry(surfpos_vocab))

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
    with open(args.modelname+'/pos_vocab.json', 'w') as file:
        file.write(json.dumps(pos_vocab, ensure_ascii=False))
    with open(args.modelname+'/polar_vocab.json', 'w') as file:
        file.write(json.dumps(polar_vocab, ensure_ascii=False))
    with open(args.modelname+'/tense_vocab.json', 'w') as file:
        file.write(json.dumps(tense_vocab, ensure_ascii=False))
    with open(args.modelname+'/surfpos_vocab.json', 'w') as file:
        file.write(json.dumps(surfpos_vocab, ensure_ascii=False))
    return  data, vocab, (surf_data_str, feat_data_str)
    
def read_valdata(maxdsize, file, vocab, mode):
    surface_vocab, feature_vocab, pos_vocab, polar_vocab, tense_vocab, surfpos_vocab = vocab
    surf_data = []; feat_data = []; pos_data = []; root_data = [];  data = []; polar_data = []; tense_data = []; surfpos_data = []
    surf_data_str = []; feat_data_str = []; root_data_str = []
    count = 0
    with open(file, 'r') as reader:
        for line in reader:   
            count += 1
            if count > maxdsize:
                break
            split_line = line.split()
            root_str = split_line[1].replace('^','+').split('+')[0]
            pos_tag  = split_line[1].replace('^','+').split('+')[1]
            tags     = split_line[1].replace('^','+').split('+')[1:]
            surf_data_str.append(split_line[0]) # add verbal versions
            root_data_str.append(root_str)      # add verbal versions
            surf_data.append([surface_vocab[char] for char in split_line[0]])
            root = [char for char in root_str]
            root_and_tags =  root + tags
            root_data.append([surface_vocab[char] for char in root])
            feat_data.append([feature_vocab[tag] for tag in root_and_tags])
            pos_data.append(pos_vocab[pos_tag])

            # fill polar data
            polars = ['Neg', 'Pos']
            polar_info_exists = False
            for tag in reversed(tags):
                if tag in polars:
                    polar_data.append(polar_vocab[tag])
                    polar_info_exists = True
                    break
            if not polar_info_exists:
                polar_data.append(polar_vocab['<unk>'])

            # fill tense data
            tenses = ['Past', 'Narr', 'Fut', 'Prog1', 'Prog2', 'Aor']
            tense_info_exists = False
            for tag in reversed(tags):
                if tag in tenses:
                    tense_data.append(tense_vocab[tag])
                    tense_info_exists = True
                    break
            if not tense_info_exists:
                tense_data.append(tense_vocab['<unk>'])

            # fill surfpos data
            #surfpostags = ['Verb', 'Noun', 'Adj', 'Ques']
            surfpostags = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Ques', 'Num', 'Postp', 'Conj', 'Dup', 'Interj', 'Det'] 
            for tag in reversed(tags):
                if tag in surfpostags:
                    surfpos_data.append(surfpos_vocab[tag])
                    break

    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('feat_data:',  len(feat_data))
    print('root_data:',  len(root_data))
    print('pos_data:',   len(pos_data))
    print('polar_data:', len(polar_data))
    print('tense_data:', len(tense_data))
    print('surfpos_data:', len(surfpos_data))
    assert len(surf_data) == len(feat_data) == len(root_data) == len(pos_data) == len(polar_data) == len(tense_data) == len(surfpos_data) 
    for (surf, feat, pos, root, polar, tense, surfpos) in zip(surf_data, feat_data, pos_data, root_data, polar_data, tense_data, surfpos_data):
        data.append([surf, feat, [pos], root, [polar], [tense], [surfpos]])
    return data
    
def get_batch(x, vocab):
    global number_of_feat_tokens, number_of_feat_unks, number_of_surf_tokens, number_of_surf_unks

    surface_vocab, feature_vocab, pos_vocab, polar_vocab, tense_vocab, surfpos_vocab = vocab

    surf, feat, pos, root, polar, tense, surfpos = [], [], [], [], [], [], [] 

    max_surf_len = max([len(s[0]) for s in x])
    max_feat_len = max([len(s[1]) for s in x])
    max_root_len = max([len(s[3]) for s in x])

    for surf_idx, feat_idx, pos_idx, root_idx, polar_idx, tense_idx, surfpos_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)

        feat_padding = [feature_vocab['<pad>']] * (max_feat_len - len(feat_idx)) 
        feat.append([feature_vocab['<s>']] + feat_idx + [feature_vocab['</s>']] + feat_padding)

        root_padding = [surface_vocab['<pad>']] * (max_root_len - len(root_idx)) 
        root.append([surface_vocab['<s>']] + root_idx + [surface_vocab['</s>']] + root_padding)

        pos.append(pos_idx)
        polar.append(polar_idx)
        tense.append(tense_idx)
        surfpos.append(surfpos_idx)

        # Count statistics...
        number_of_feat_tokens += len(feat_idx)
        number_of_feat_unks += feat_idx.count(feature_vocab['<unk>'])
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])

    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(feat, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(pos,  dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(root, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(polar, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(tense, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(surfpos, dtype=torch.long, requires_grad=False, device=device)

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
    tst_batches, _ = get_batches(tstdata, vocab, args.batchsize, args.seq_to_no_pad) 
    #tstdata = None; tst_batches = None

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
