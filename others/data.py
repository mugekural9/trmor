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

        # Probe 'trmor_data/surf/surf.uniquesurfs.trn.txt'
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
                #data.append([vocab[word] for word in split_line])
                data.append([vocab[char] for char in split_line[0]])
                #data.append([vocab[char] for char in split_line[1]])
       
        '''with open(fname) as fin:
            for line in fin:
                if '#' in line or line == '\n':
                    continue
                split_line = line.split('\t') #line.split()
                if len(split_line) < 1:
                    dropped += 1
                    continue

                if max_length:
                    if len(split_line) > max_length:
                        dropped += 1
                        continue
                data.append([vocab[char] for char in split_line[1]])'''
            
        '''with open(fname) as fin:
            for line in fin:
                if line == '\n':
                    continue
                surfs = line.strip().split(' ')
                if len(surfs) < 1:
                    dropped += 1
                    continue
                if max_length:
                    if len(surfs) > max_length:
                        dropped += 1
                        continue
                for surf in surfs:
                  data.append([vocab[token] for token in surf])'''
        
        # morpho challenge 2005 - unsupervised morpheme segmentation
        '''with open(fname) as fin:
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
                data.append([vocab[char] for char in split_line[0].strip().split(' ')[1]])''' 


        if isinstance(vocab, VocabEntry):
            return data, vocab, dropped, labels

        return data, VocabEntry(vocab), dropped, labels
