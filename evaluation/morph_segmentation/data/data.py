import re, torch, json, os
from collections import defaultdict, Counter
from common.vocab import VocabEntry
from common.batchify import get_batches


def read_data(maxdsize, file, surface_vocab, mode):
    surf_data = []; data = []
    all_surfs = dict()
    count = 0
    with open(file, 'r') as reader:
        for line in reader: 
            count += 1
            #if count < 5:
            #    continue
            if count > maxdsize:
                break
            surf = line.strip().split('\t')[0]
            #if len(surf) > 10:
            #    continue
            surf_data.append([surface_vocab[char] for char in surf])
            
    print(mode,':')
    print('surf_data:',  len(surf_data))
    for surf in surf_data:
        data.append([surf])
    return data
    
def build_data(args):
    # Read data and get batches...
    tst_data = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')
    tst_batches, _ = get_batches(tst_data, args.vocab, args.batch_size, '') 
    return tst_data, tst_batches


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
                data.append([vocab[char] for char in split_line[0].strip().split(' ')[0]])


        if isinstance(vocab, VocabEntry):
            return data, vocab, dropped, labels

        return data, VocabEntry(vocab), dropped, labels

