import re, torch, json, os
from sys import breakpointhook
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches



#UNISCHEMA_tags = dict()
#with open('/home/mugekural/dev/git/trmor/model/vqvae/data/universal-schema', 'r') as reader:
#    for line in reader:
#        key =line.strip().split(' ')[0]
#        tag =line.strip().split(' ')[-1]
#        UNISCHEMA_tags[tag] = key
#print(UNISCHEMA_tags)
with open('UNISCHEMA_tags.json','r') as f:
    UNISCHEMA_tags = json.load(f)
#with open('UNISCHEMA_tags.json', 'w') as f:
#    f.write(json.dumps(UNISCHEMA_tags))


def get_language_tags(maxdsize, file):
    tag_vocabs = dict()
    tag_data = dict()
    count=0
    
    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            lemma,inflected_surf, tags = line.strip().lower().split('\t')
            added_tagnames = []
            for label in tags.split(';'):
                label = label.lower()
                tagname = UNISCHEMA_tags[label]
                added_tagnames.append(tagname)
                if tagname not in tag_vocabs:
                    tag_vocabs[tagname] = {
                        '<pad>' :0
                    }

    return tag_vocabs



def read_data(maxdsize, file, surface_vocab, tag_vocabs, mode):
    lemma_data = []; data = []; 
    inflected_surf_data = []
    
    count = 0
    lemmas  = []; inflected_surfs = []
   
    tag_data = dict()
    for key,vals in tag_vocabs.items():
        tag_data[key] = []

    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            lemma,inflected_surf, tags = line.strip().lower().split('\t')
            added_tagnames = []
            for label in tags.split(';'):
                label = label.lower()
                tagname = UNISCHEMA_tags[label]
                added_tagnames.append(tagname)

                if mode=='PREPAREDATA':
                    if label not in tag_vocabs[tagname]:
                        tag_vocabs[tagname][label] = len(tag_vocabs[tagname])
                tag_data[tagname].append([tag_vocabs[tagname][label]])
            for tname in tag_vocabs.keys():
                if tname not in added_tagnames:
                    tag_data[tname].append([tag_vocabs[tname]["<pad>"]])
            lemma_data.append([surface_vocab[char] for char in lemma])
            inflected_surf_data.append([surface_vocab[char] for char in inflected_surf])
            lemmas.append(lemma)
            inflected_surfs.append(inflected_surf)
    print(mode + ':')
    print('\nlemma_data:' +  str(len(lemma_data)))
    print('\nreinflected_surf_data:' +  str(len(inflected_surf_data)))
    print('\ntag_data:'  +  str(len(tag_data)))
    
    for j in range(len(lemma_data)):
        instance= []
        instance.append(lemma_data[j])
        tagkeys = []
        for key in tag_data.keys():
            instance.append(tag_data[key][j])
            tagkeys.append(key)
        instance.append(inflected_surf_data[j])
        #data.append(instance)
        data.append((instance,['lemma']+tagkeys+['inflected_surf']))
    
    
    return data, tag_vocabs


##workaround!
def read_data_unsup(maxdsize, file, surface_vocab, mode):
    surf_data = []; data = []; 
    count = 0
    surfs  = []; 
    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            #surf = line.strip()
            lemma,inflected_surf, tags = line.strip().lower().split('\t')

            surf_data.append([surface_vocab[char] for char in inflected_surf])
            surfs.append(inflected_surf)
            surf_data.append([surface_vocab[char] for char in lemma])
            surfs.append(lemma)
    print(mode + ':')
    print('\nsurf_data:' +  str(len(surf_data)))
    for surf in surf_data:
        data.append([surf])
    return data


def build_data(args, surface_vocab=None):
    # Read data and get batches...
    init_tag_vocabs = get_language_tags(args.maxtrnsize, args.trndata)
    trndata, tag_vocabs = read_data(args.maxtrnsize, args.trndata, surface_vocab, init_tag_vocabs, 'PREPAREDATA')
    args.trnsize = len(trndata)
    lxsrc_ordered_batches, _ = get_batches_msved(trndata, surface_vocab, tag_vocabs, args.batchsize, args.seq_to_no_pad) 

    lxtgtdata, _ = read_data(args.maxtrnsize, args.trndata, surface_vocab, tag_vocabs, 'TRN')

    args.lxtgtsize = len(lxtgtdata)
    lxtgt_ordered_batches, _ = get_batches_msved(lxtgtdata, surface_vocab, tag_vocabs, args.batchsize, 'feature')

    lxtgtdata, _ = read_data(args.maxtrnsize, args.tstdata, surface_vocab, tag_vocabs, 'LXTGT_ORDERED_TST')  
    lxtgt_ordered_batches_TST, _ = get_batches_msved(lxtgtdata, surface_vocab, tag_vocabs, 1, 'feature')


    valdata, _ = read_data(args.maxvalsize, args.valdata, surface_vocab, tag_vocabs, 'TRN') 
    val_batches, _ = get_batches_msved(valdata, surface_vocab, tag_vocabs, args.batchsize, 'feature')

    tstdata, _ = read_data(args.maxtstsize, args.tstdata, surface_vocab, tag_vocabs, 'TST')  
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches_msved(tstdata, surface_vocab, tag_vocabs, 1, args.seq_to_no_pad) 
    
    udata = read_data_unsup(args.maxusize, args.unlabeled_data, surface_vocab, 'UDATA')
    u_batches, _ = get_batches(udata, surface_vocab, args.batchsize, args.seq_to_no_pad) 
    return (trndata, valdata, tstdata, udata), (lxsrc_ordered_batches, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, val_batches, tst_batches, u_batches), surface_vocab, tag_vocabs


## Data prep
def get_batch_tagmapping(x, surface_vocab, tag_vocabs, device='cuda'):
    global number_of_surf_tokens, number_of_surf_unks
    lemma =[];  inflect_surf = [] 
    max_lemma_len = max([len(s[0][0]) for s in x])
    max_inflect_surf_len = max([len(s[0][-1]) for s in x])

    #['Part', 'Case', 'Number', 'Possession', 'Mood', 'Person', 'Tense', 'Aspect', 'Polarity', 'Interrogativity', 'Language-Specific', 'Valency']
    
    datadict= dict()
    for key,val in tag_vocabs.items():
        datadict[key] = []
    
    
    
    for instance in x:
        for m, _idx in zip(instance[1], instance[0]):
            if m =='lemma':
                lemma_idx= _idx
                lemma_padding = [surface_vocab['<pad>']] * (max_lemma_len - len(lemma_idx)) 
                lemma.append([surface_vocab['<s>']] + lemma_idx + [surface_vocab['</s>']] + lemma_padding)
            elif m=='inflected_surf':
                inflect_surf_idx = _idx
                inflect_surf_padding = [surface_vocab['<pad>']] * (max_inflect_surf_len - len(inflect_surf_idx)) 
                inflect_surf.append([surface_vocab['<s>']] + inflect_surf_idx + [surface_vocab['</s>']] + inflect_surf_padding)
            else:
                datadict[m].append(_idx)

    
    
        # Count statistics...
        number_of_surf_tokens += len(lemma_idx)
        number_of_surf_unks += lemma_idx.count(surface_vocab['<unk>'])
    
    tagtensors =[]
    for key,val in datadict.items():
        tagtensors.append(torch.tensor(val, dtype=torch.long, requires_grad=False, device=device))
    return  torch.tensor(lemma, dtype=torch.long,  requires_grad=False, device=device), tagtensors, \
            torch.tensor(inflect_surf, dtype=torch.long, requires_grad=False, device=device)

   

def get_batches_msved(data, vocab, tag_vocabs, batchsize=64, seq_to_no_pad='', device='cuda'):
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
            z = sorted(zip(order, data), key=lambda i: len(i[1][0][0]))
        if seq_to_no_pad == 'feature':
            z = sorted(zip(order, data), key=lambda i: len(i[1][0][-1]))

    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
        if not continuity:
            jr = i
            # data (surfaceform, featureform)
            if seq_to_no_pad == 'surface':
                while jr < min(len(data), i+batchsize) and len(data[jr][0][0]) == len(data[i][0][0]): # Do not pad and select equal length of, 0: +surfaceform, 1: +featureform, 2:rootpostag, 3: +root
                    jr += 1
            elif seq_to_no_pad == 'feature':
                while jr < min(len(data), i+batchsize) and len(data[jr][0][-1]) == len(data[i][0][-1]): 
                    jr += 1
            elif seq_to_no_pad == 'surface+feature':
                while jr < min(len(data), i+batchsize) and len(data[jr][0][0]) == len(data[i][0][0]) and len(data[jr][0][-1]) == len(data[i][0][-1]): 
                    jr += 1
            batches.append(get_batch_tagmapping(data[i: jr], vocab, tag_vocabs, device=device))
            i = jr
        else:
            batches.append(get_batch_tagmapping(data[i: i+batchsize], vocab, tag_vocabs, device=device))
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

