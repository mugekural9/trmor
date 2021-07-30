import re
import torch
import json
from data import Vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
datamap= dict()
surfacemap = dict()

def readdata():
    snts = dict()
    data = []
    with open('trmor_data/trmor2018.train', 'r') as reader:
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

    vocab_file = 'trmor_data/src_vocab.txt'    
    Vocab.build(data, vocab_file, 10000)
    vocab = Vocab(vocab_file)
    with open('trmor_data/src_vocab.json', 'w') as file:
        file.write(json.dumps(vocab.word2id)) 
    return data, vocab
    
def get_batch(x, vocab, tgt_vocab):
    src, tgt = [], []
    max_len = max([len(s[1]) for s in x])
    max_tgt_len = max([len(s[0]) for s in x])
    for snt in x:
        input  = snt[1]
        target = snt[0]
        target = [char for char in target]        
        
        s_idx =  [vocab.word2id[w] if w in vocab.word2id else vocab.unk for w in input]  
        t_idx =  [tgt_vocab.word2id[w] if w in tgt_vocab.word2id else tgt_vocab['<unk>'] for w in target]
        
        padding = [vocab.pad] * (max_len - len(input)) 
        src.append([vocab.go] + s_idx + [vocab.eos] + padding)
        
        tgt_padding = [tgt_vocab['<pad>']] * (max_tgt_len - len(target)) 
        tgt.append([tgt_vocab['<s>']] + t_idx + [tgt_vocab['</s>']] + tgt_padding)

    return torch.LongTensor(src).t().contiguous().to(device), \
        torch.LongTensor(tgt).t().contiguous().to(device)  # time * batch

def get_batches(data, vocab, tgt_vocab, batchsize=64):
    order = range(len(data))
    # 0:sort according to surfaceform, 1: featureform 
    z = sorted(zip(order, data), key=lambda i: len(i[1][1]))
    # z = zip(order, data)
    order, data = zip(*z)
    batches = []
    i = 0
    while i < len(data):
        j = i
        # form batch with equal length of tgt
        # data (surfaceform, featureform)
        while j < min(len(data), i+batchsize) and len(data[j][1]) == len(data[i][1]): # 0 esitligi: surfaceform esitligi, 1 esitligi: surfaceform+featureform esitligi
            j += 1
        batches.append(get_batch(data[i: j], vocab, tgt_vocab))
        i = j
    return batches, order    
