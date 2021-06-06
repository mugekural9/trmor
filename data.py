import re
import torch
import json
from vocab import Vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict()
surfacemap = dict()

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
                splt_line = line.lower().split()
                if len(splt_line) > 2: # multiple data 
                    for i in range(1,len(splt_line)):
                        if splt_line[i] not in datamap and splt_line[0] not in surfacemap:
                            surfacemap[splt_line[0]] = 'here'
                            datamap[splt_line[i]] = splt_line[0]
                            tags = splt_line[i].replace('^','+').split('+') 
                            tags = [char for char in tags[0]] + tags[1:]
                            snt.append([splt_line[0], tags])
                else:
                    if splt_line[1] not in datamap  and splt_line[0] not in surfacemap:
                        surfacemap[splt_line[0]] = 'here'
                        datamap[splt_line[1]] = splt_line[0]
                        tags = splt_line[1].replace('^','+').split('+')
                        tags = [char for char in tags[0]] + tags[1:]
                        snt.append([splt_line[0], tags])

    vocab_file = 'vocab2.txt'    
    Vocab.build(data, vocab_file, 10000)
    vocab = Vocab(vocab_file)
    with open('vocab.json', 'w') as file:
        file.write(json.dumps(vocab.word2idx)) 
    return data, vocab
    
def get_batch(x, vocab):
    go_x, x_eos = [], []
    max_len = max([len(s[1]) for s in x])
    max_tgt_len = max([len(s[0]) for s in x])
    for snt in x:
        input  = snt[1]
        target = snt[0]
        target = [char for char in target]        
        s_idx =  [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in input]  
        t_idx =  [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in target]
        padding = [vocab.pad] * (max_len - len(input)) 
        go_x.append([vocab.go] + s_idx + [vocab.eos] + padding)
        padding = [vocab.pad] * (max_tgt_len - len(target)) 
        x_eos.append([vocab.go] + t_idx + [vocab.eos] + padding)

    #sents_ts = torch.tensor(x_eos, dtype=torch.long,
    #                             requires_grad=False, device=device)

    #return sents_ts
    
    return torch.LongTensor(go_x).t().contiguous().to(device), \
        torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, batchsize=64):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batchsize) and len(data[j][0]) == len(data[i][0]):
            j += 1
        batches.append(get_batch(data[i: j], vocab))
        i = j
    return batches, order    
