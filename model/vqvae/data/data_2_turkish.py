import re, torch, json, os
from sys import breakpointhook
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches

def read_data(maxdsize, file, surface_vocab, mode, tag_vocabs=None):
    surf_data = []; data = []; tag_data = dict(); 
    reinflected_surf_data = []

    if tag_vocabs is None:
        tag_vocabs = dict()

        case_vocab = defaultdict(lambda: len(case_vocab))
        case_vocab['<pad>'] = 0

        polar_vocab = defaultdict(lambda: len(polar_vocab))
        polar_vocab['<pad>'] = 0

        mood_vocab = defaultdict(lambda: len(mood_vocab))
        mood_vocab['<pad>'] = 0
        
        evid_vocab = defaultdict(lambda: len(evid_vocab))
        evid_vocab['<pad>'] = 0

        pos_vocab = defaultdict(lambda: len(pos_vocab))
        pos_vocab['<pad>'] = 0

        per_vocab = defaultdict(lambda: len(per_vocab))
        per_vocab['<pad>'] = 0

        num_vocab = defaultdict(lambda: len(num_vocab))
        num_vocab['<pad>'] = 0

        tense_vocab = defaultdict(lambda: len(tense_vocab))
        tense_vocab['<pad>'] = 0

        aspect_vocab = defaultdict(lambda: len(aspect_vocab))
        aspect_vocab['<pad>'] = 0

        inter_vocab = defaultdict(lambda: len(inter_vocab))
        inter_vocab['<pad>'] = 0

        poss_vocab = defaultdict(lambda: len(poss_vocab))
        poss_vocab['<pad>'] = 0
    

        tag_vocabs['case'] = case_vocab

        tag_vocabs['polar'] = polar_vocab

        tag_vocabs['mood'] = mood_vocab

        tag_vocabs['evid'] = evid_vocab

        tag_vocabs['pos'] = pos_vocab

        tag_vocabs['per'] = per_vocab

        tag_vocabs['num'] = num_vocab

        tag_vocabs['tense'] = tense_vocab

        tag_vocabs['aspect'] = aspect_vocab

        tag_vocabs['inter'] = inter_vocab
        tag_vocabs['poss'] = poss_vocab
    
    tag_data['case'] = []
    tag_data['polar'] = []
    tag_data['mood'] = []
    tag_data['evid'] = []
    tag_data['pos'] = []
    tag_data['per'] = []
    tag_data['num'] = []
    tag_data['tense'] = []
    tag_data['aspect'] = []
    tag_data['inter'] = []
    tag_data['poss'] = []

        
    
    count = 0
    surfs  = []; reinflected_surfs = []
   
    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            surf,tags,reinflected_surf = line.strip().split('\t')
            added_tagnames = []
            for z in tags.split(','):
                tagname, label = z.split('=')
                added_tagnames.append(tagname)
                tag_data[tagname].append([tag_vocabs[tagname][label]])
            for tname in tag_vocabs.keys():
                if tname not in added_tagnames:
                    tag_data[tname].append([tag_vocabs[tname]["<pad>"]])
            surf_data.append([surface_vocab[char] for char in surf])
            reinflected_surf_data.append([surface_vocab[char] for char in reinflected_surf])
            surfs.append(surf)
            reinflected_surfs.append(reinflected_surf)
    print(mode + ':')
    print('\nsurf_data:' +  str(len(surf_data)))
    print('\nreinflected_surf_data:' +  str(len(reinflected_surf_data)))

    print('\ntag_data:'  +  str(len(tag_data)))
    for surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, reinflected_surf in zip(surf_data, tag_data['case'], tag_data['polar'],tag_data['mood'],tag_data['evid'],tag_data['pos'],tag_data['per'],tag_data['num'],tag_data['tense'],tag_data['aspect'],tag_data['inter'],tag_data['poss'], reinflected_surf_data):
        data.append([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, reinflected_surf])
    return data, tag_vocabs


def read_data_unsup(maxdsize, file, surface_vocab, mode):
    surf_data = []; data = []; 
    count = 0
    surfs  = []; 
    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            surf = line.strip()
            surf_data.append([surface_vocab[char] for char in surf])
            surfs.append(surf)
    print(mode + ':')
    print('\nsurf_data:' +  str(len(surf_data)))
    for surf in surf_data:
        data.append([surf])
    return data


def build_data(args, surface_vocab=None, tag_vocabs=None):
    # Read data and get batches...
    trndata, tag_vocabs = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'TRN', tag_vocabs)
    args.trnsize = len(trndata)
    lxsrc_ordered_batches, _ = get_batches_msved(trndata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    lxtgtdata, _ = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'TRN', tag_vocabs)    
    args.lxtgtsize = len(lxtgtdata)
    lxtgt_ordered_batches, _ = get_batches_msved(lxtgtdata, surface_vocab, args.batchsize, 'feature')

    lxtgtdata, _ = read_data(args.maxtstsize, args.tstdata, surface_vocab, 'LXTGT_ORDERED_TST', tag_vocabs)   
    lxtgt_ordered_batches_TST, _ = get_batches_msved(lxtgtdata, surface_vocab, 1, 'feature')


    valdata, _ = read_data(args.maxvalsize, args.valdata, surface_vocab, 'TRN',tag_vocabs)    
    args.valsize = len(valdata)
    val_batches, _ = get_batches_msved(valdata, surface_vocab, args.batchsize, 'feature')


    tstdata, _ = read_data(args.maxtstsize, args.tstdata, surface_vocab, 'TST', tag_vocabs)
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches_msved(tstdata, surface_vocab, 1, args.seq_to_no_pad) 
    
    udata = read_data_unsup(args.maxtrnsize, args.unlabeled_data, surface_vocab, 'UDATA')
    u_batches, _ = get_batches(udata, surface_vocab, args.batchsize, '') 

    return (trndata, valdata, tstdata, udata), (lxsrc_ordered_batches, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, val_batches, tst_batches, u_batches), surface_vocab, tag_vocabs


## Data prep
def get_batch_tagmapping(x, surface_vocab, device='cuda'):
    global number_of_surf_tokens, number_of_surf_unks
    surf =[]; case=[]; polar =[]; mood=[]; evid=[]; pos=[]; per=[]; num=[]; tense=[]; aspect=[]; inter=[]; poss=[]; reinflect_surf = [] 
    max_surf_len = max([len(s[0]) for s in x])
    max_reinflect_surf_len = max([len(s[-1]) for s in x])
    for surf_idx, case_idx,polar_idx, mood_idx ,evid_idx,pos_idx,per_idx,num_idx,tense_idx,aspect_idx,inter_idx,poss_idx, reinflect_surf_idx  in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        reinflect_surf_padding = [surface_vocab['<pad>']] * (max_reinflect_surf_len - len(reinflect_surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        reinflect_surf.append([surface_vocab['<s>']] + reinflect_surf_idx + [surface_vocab['</s>']] + reinflect_surf_padding)
        case.append(case_idx)
        polar.append(polar_idx)
        mood.append(mood_idx)
        evid.append(evid_idx)
        pos.append(pos_idx)
        per.append(per_idx)
        num.append(num_idx)
        tense.append(tense_idx)
        aspect.append(aspect_idx)
        inter.append(inter_idx)
        poss.append(poss_idx)
        # Count statistics...
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
    
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            [torch.tensor(case, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(polar, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(mood, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(evid, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(pos, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(per, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(num, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(tense, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(aspect, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(inter, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(poss, dtype=torch.long, requires_grad=False, device=device)], \
            torch.tensor(reinflect_surf, dtype=torch.long, requires_grad=False, device=device)

def get_batches_msved(data, vocab, batchsize=64, seq_to_no_pad='', device='cuda'):
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
        if seq_to_no_pad == 'feature':
            z = sorted(zip(order, data), key=lambda i: len(i[1][-1]))

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
                while jr < min(len(data), i+batchsize) and len(data[jr][-1]) == len(data[i][-1]): 
                    jr += 1
            elif seq_to_no_pad == 'surface+feature':
                while jr < min(len(data), i+batchsize) and len(data[jr][0]) == len(data[i][0]) and len(data[jr][-1]) == len(data[i][-1]): 
                    jr += 1
            batches.append(get_batch_tagmapping(data[i: jr], vocab, device=device))
            i = jr
        else:
            batches.append(get_batch_tagmapping(data[i: i+batchsize], vocab, device=device))
            i += batchsize
    #args.logger.write('# of surf tokens: ' + number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    return batches, order   

'''class MonoTextData(object):
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

        if 'unique' in fname:
            with open(fname) as fin:
                for line in fin:
                    split_line = line.split('\t') #line.split()
                    if len(split_line) < 1:
                        dropped += 1
                        continue

                    if max_length:
                        if len(split_line) > max_length:
                            dropped += 1
                            continue
                    data.append([vocab[char] for char in split_line[0]])
        elif '.tur' in fname:
            with open(fname) as fin:
                for line in fin:
                    if label:
                        split_line = line.split('\t')
                        lb = split_line[0]
                        split_line = split_line[1].split()
                    else:
                        split_line = line.split('\t') #line.split()
                    if len(split_line) < 1:
                        dropped += 1
                        continue
                    if max_length:
                        if len(split_line) > max_length:
                            dropped += 1
                            continue
                    if label:
                        labels.append(lb)
                    data.append([vocab[char] for char in split_line[0].strip().split(' ')[1]])
        elif 'goldstdsample.tur' in fname:
            with open(fname) as fin:
                for line in fin:
                    if label:
                        split_line = line.split('\t')
                        lb = split_line[0]
                        split_line = split_line[1].split()
                    else:
                        split_line = line.split('\t') #line.split()
                    if len(split_line) < 1:
                        dropped += 1
                        continue
                    if max_length:
                        if len(split_line) > max_length:
                            dropped += 1
                            continue
                    if label:
                        labels.append(lb)
                    data.append([vocab[char] for char in split_line[0]])
        elif 'turkish' in fname:
            with open(fname) as fin:
                for line in fin:
                    split_line = line.strip().split('\t')
                    if len(split_line) < 1:
                        dropped += 1
                        continue
                    if max_length:
                        if len(split_line) > max_length:
                            dropped += 1
                            continue
                    data.append([vocab[char] for char in split_line[0]])
        elif 'zhou' in fname:
            with open(fname) as fin:
                for line in fin:
                    split_line = line.strip().split('\t')
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

        return data, VocabEntry(vocab), dropped, labels'''

