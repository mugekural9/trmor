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
    
    with open(file, 'r') as reader:
        for line in reader:     
            count += 1
            if count > maxdsize:
                break
            surf = line.strip().lower().split('\t')[1]
            surf_data.append([surface_vocab[char] for char in surf])
            surfs.append(surf)

    print(mode,':')
    print('surf_data:',  len(surf_data))
    for surf in surf_data:
        data.append([surf])
    return data
    
def build_data(args, surface_vocab=None, mode='none'):
    # Read data and get batches...
    if mode=='pretrain_ae':
        surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
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
                split_line = line.strip().lower().split('\t')
                if len(split_line) < 1:
                    dropped += 1
                    continue
                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue
                data.append([vocab[char] for char in split_line[1]])


        if isinstance(vocab, VocabEntry):
            return data, vocab, dropped, labels

        return data, VocabEntry(vocab), dropped, labels

