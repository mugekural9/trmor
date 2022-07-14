import re, torch, json, os
from sys import breakpointhook
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches


def read_data(maxdsize, file, surface_vocab, mode):
    surf_data = []; data = []
    all_surfs = dict()
    count = 0
    surfs  = []
    if 'unique' in file:
        with open(file, 'r') as reader:
            for line in reader: 
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split('\t')[0]
                surf_data.append([surface_vocab[char] for char in surf])
    elif '.tur' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split(' ')[1]
                surf_data.append([surface_vocab[char] for char in surf])
    elif 'task3' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split('\t')[2]#[0]
                #if surf not in surfs:
                surf_data.append([surface_vocab[char] for char in surf])
                surfs.append(surf)
                #surf_r = surf[::-1]
                #surf_data.append([surface_vocab[char] for char in surf_r])
    elif 'zhou' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split('\t')[0]
                #if surf not in surfs:
                surf_data.append([surface_vocab[char] for char in surf])
                surfs.append(surf)
    elif 'trn.txt' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf = line.strip().split('\t')[0]
                if surf not in surfs:
                    surf_data.append([surface_vocab[char] for char in surf])
                    surfs.append(surf)
                #surf_r = surf[::-1]
                #surf_data.append([surface_vocab[char] for char in surf_r])
    print(mode,':')
    print('surf_data:',  len(surf_data))
    for surf in surf_data:
        data.append([surf])
    return data
    
def build_data(args, surface_vocab=None):
    # Read data and get batches...
    #surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
    trndata = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'TRN')
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata = read_data(args.maxvalsize, args.valdata, surface_vocab, 'VAL')    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata = read_data(args.maxtstsize, args.tstdata, surface_vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, surface_vocab, args.batchsize, '') 
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
        elif 'task3' in fname:
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

