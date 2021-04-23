import re
import torch
import json
from vocab import Vocab

def readdata():
    snts = dict()
    data = []
    with open('trmor2018.train', 'r') as reader:
        # Read and print the entire file line by line
        snt = []
        sent_state = False
        for line in reader:        
            if '<S id' in line:
                s_id = re.search('id="(.*)"', line).group(1) 
                sent_state = True
                snt = []
                continue
            elif "</S>" in line:
                snts[s_id] = snt
                data.extend(snt)
                sent_state = False
                continue
            if sent_state:
                splt_line = line.split()
                if len(splt_line) > 2: # multiple data 
                    for i in range(1,len(splt_line)):
                        tags = splt_line[i].split('+') 
                        tags = [char for char in tags[0]] + tags[1:]
                        snt.append([splt_line[0], tags])
                else:
                    tags = splt_line[1].split('+')
                    tags = [char for char in tags[0]] + tags[1:]
                    snt.append([splt_line[0], tags])

    vocab_file = 'vocab.txt'    
    Vocab.build(data, vocab_file, 10000)
    vocab = Vocab(vocab_file)
    # with open('vocab.json', 'w') as file:
    #      file.write(json.dumps(vocab.word2idx)) 
    return data, vocab
    
def get_batch(x, vocab):
    go_x, x_eos = [], []
    max_len = max([len(s[1]) for s in x])
    max_tgt_len = max([len(s[0]) for s in x])
    for snt in x:
        input  = snt[1]
        target = snt[0]
        target = [char for char in target]        
        snt = [char for char in snt[0]] + snt[1]
        s_idx =  [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in input]  
        t_idx =  [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in target]
        padding = [vocab.pad] * (max_len - len(input)) 
        go_x.append([vocab.go] + s_idx + [vocab.eos] + padding)
        padding = [vocab.pad] * (max_tgt_len - len(target)) 
        x_eos.append([vocab.go] + t_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous(), \
        torch.LongTensor(x_eos).t().contiguous()  # time * batch

def get_batches(data, vocab, batchsize=64):
        order = range(len(data))
        z = sorted(zip(order, data), key=lambda i: len(i[1]))
        order, data = zip(*z)
        batches = []
        i = 0
        while i < len(data):
            j = i
            while j < min(len(data), i+batchsize) and len(data[j]) == len(data[i]):
                j += 1
            batches.append(get_batch(data[i: j], vocab))
            i = j
        return batches, order        
