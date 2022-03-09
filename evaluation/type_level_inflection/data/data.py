import re, torch, json, os
from common.vocab import  VocabEntry
from collections import defaultdict

number_of_surf_tokens = 0; number_of_surf_unks = 0
number_of_feat_tokens = 0; number_of_feat_unks = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def read_trndata_makevocab(args, surf_vocab):
    data = []; surf_data = []; feat_data = []
    with open(args.trndata, 'r') as reader:
        feat_vocab =  defaultdict(lambda: len(feat_vocab))
        feat_vocab['<pad>'] = 0
        feat_vocab['<s>'] = 1
        feat_vocab['</s>'] = 2
        feat_vocab['<unk>'] = 3
        count = 0
        for line in reader: 
            count += 1
            if count > args.maxtrnsize:
                break
            split_line = line.split()
            tags = split_line[1].replace('^','+').split('+')
            lemma = list(tags[0])
            tags = lemma + tags[1:]
            # fill surface vocab and data
            surf_data.append([surf_vocab[char] for char in split_line[0]])
            # fill feature vocab and data
            feat_data.append([feat_vocab[tag] for tag in tags])
    print('TRN:')
    print('surf_data:',  len(surf_data))
    print('feat_data:', len(feat_data))

    assert len(surf_data) == len(feat_data)
    for (surf, feat) in zip(surf_data,  feat_data):
        data.append([surf, feat])
    return  data, VocabEntry(feat_vocab)
    
def read_valdata(maxdsize, file, vocab, mode):
    surf_vocab, feat_vocab = vocab
    data = []; surf_data = []; feat_data = []
    count = 0
    with open(file, 'r') as reader:
        for line in reader:   
            count += 1
            if count > maxdsize:
                break
            split_line = line.split()
            tags = split_line[1].replace('^','+').split('+')
            lemma = list(tags[0])
            tags = lemma + tags[1:]
            # fill surface vocab and data
            surf_data.append([surf_vocab[char] for char in split_line[0]])
            # fill feature vocab and data
            feat_data.append([feat_vocab[tag] for tag in tags])
    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('feat_data:', len(feat_data))
    assert len(surf_data) == len(feat_data) 
    for (surf, feat) in zip(surf_data, feat_data):
        data.append([surf, feat])
    return data
    
def get_batch(x, vocab):
    global number_of_surf_tokens, number_of_feat_tokens, number_of_surf_unks, number_of_feat_unks
    surface_vocab,  feature_vocab = vocab
    surf, feat = [], []
    max_surf_len = max([len(s[0]) for s in x])
    max_feat_len = max([len(s[1]) for s in x])
    for surf_idx, feat_idx in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        feat_padding = [feature_vocab['<pad>']] * (max_feat_len - len(feat_idx)) 
        feat.append([feature_vocab['<s>']] + feat_idx + [feature_vocab['</s>']] + feat_padding)
        # Count statistics...
        number_of_surf_tokens += len(surf_idx);  number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
        number_of_feat_tokens += len(feat_idx);  number_of_feat_unks += feat_idx.count(feature_vocab['<unk>'])
    
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(feat, dtype=torch.long, requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad=''):
    continuity = (seq_to_no_pad == '')
    print('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
    # reset dataset statistics
    global  number_of_surf_tokens, number_of_feat_tokens, number_of_surf_unks, number_of_feat_unks
    number_of_surf_tokens = 0
    number_of_surf_unks = 0
    number_of_feat_tokens = 0
    number_of_feat_unks = 0
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
                while jr < min(len(data), i+batchsize) and len(data[jr][0]) == len(data[i][0]): # Do not pad and select equal length of, 0: +surfaceform, 1: +featureform
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
    print('# of feat tokens: ', number_of_feat_tokens, ', # of feat unks: ', number_of_feat_unks)
    
    return batches, order    

def build_data(args):
    # Read data and get batches...
    surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
    trndata, feature_vocab = read_trndata_makevocab(args, surface_vocab) 
    vocab = (surface_vocab, feature_vocab)
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_valdata(args.maxvalsize, args.valdata, vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_valdata(args.maxtstsize, args.tstdata, vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, vocab, args.batchsize, args.seq_to_no_pad) 
    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), vocab


class MonoTextData(object):
    """docstring for MonoTextData"""
    def __init__(self, fname, label=False, max_length=None, vocab=None):
        super(MonoTextData, self).__init__()
        self.data, self.vocab, self.dropped, self.labels = self._read_corpus(fname, label, max_length, vocab)

    def __len__(self):
        return len(self.data)

    def _read_corpus(self, fname, label, max_length, vocab):
        data = []
        labels = [] if label else None
        dropped = 0
        if not vocab:
            vocab = defaultdict(lambda: len(vocab))
            vocab['<pad>'] = 0
            vocab['<s>'] = 1
            vocab['</s>'] = 2
            vocab['<unk>'] = 3

        with open(fname) as fin:
            for line in fin:
                split_line = line.split('\t') 
                if len(split_line) < 1:
                    dropped += 1
                    continue
                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue
                data.append([vocab[char] for char in split_line[0]])

        if isinstance(vocab, VocabEntry):
            return data, vocab, dropped, labels

        return data, VocabEntry(vocab), dropped, labels
