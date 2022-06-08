from ast import Break
import re, torch, json, os, argparse, random
from collections import defaultdict, Counter
from common.vocab import VocabEntry
import re, torch, json, os
from common.utils import *
from vqvae import VQVAE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
number_of_surf_tokens = 0; number_of_surf_unks = 0

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = '4x10dec128_suffixd512'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/analysis/'+model_id+'/'
    args.logfile = args.logdir + '/copies.txt'
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")
    # initialize model
    # load vocab (to initialize the model with correct vocabsize)
    with open(model_vocab) as f:
        word2id = json.load(f)
        args.vocab = VocabEntry(word2id)
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.ni = 256; 
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.enc_nh = 512;
    args.dec_nh = 128#args.enc_nh; 
    args.embedding_dim = args.enc_nh; #args.nz = args.enc_nh
    args.beta = 0.5
    args.num_dicts = 4; args.nz = 512; args.outcat=0; args.incat = 512
    args.orddict_emb_num  = 10; 
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    args.model.to('cuda')
    return args

def get_batch_tagmapping(x, surface_vocab, device=device):
    global number_of_surf_tokens, number_of_surf_unks
    surf =[]; case=[]; polar =[]; mood=[]; evid=[]; pos=[]; per=[]; num=[]; tense=[]; aspect=[]; inter=[]; poss=[]; entry = [] 
    max_surf_len = max([len(s[0]) for s in x])
    for surf_idx, case_idx,polar_idx, mood_idx ,evid_idx,pos_idx,per_idx,num_idx,tense_idx,aspect_idx,inter_idx,poss_idx, entry_idx  in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
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
        entry.append(entry_idx)

        # Count statistics...
        number_of_surf_tokens += len(surf_idx)
        number_of_surf_unks += surf_idx.count(surface_vocab['<unk>'])
    return  torch.tensor(surf, dtype=torch.long,  requires_grad=False, device=device), \
            torch.tensor(case, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(polar, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(mood, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(evid, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(pos, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(per, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(num, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(tense, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(aspect, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(inter, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(poss, dtype=torch.long, requires_grad=False, device=device), \
            torch.tensor(entry, dtype=torch.long, requires_grad=False, device=device)

def get_batches(data, vocab, batchsize=64, seq_to_no_pad='', device=device):
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
    z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
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
                while jr < min(len(data), i+batchsize) and len(data[jr][1]) == len(data[i][1]): 
                    jr += 1
            batches.append(get_batch_tagmapping(data[i: jr], vocab, device=device))
            i = jr
        else:
            batches.append(get_batch_tagmapping(data[i: i+batchsize], vocab, device=device))
            i += batchsize
    print('# of surf tokens: ', number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
    return batches, order    

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
        if 'txt' in fname:
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

def read_data(maxdsize, file, surface_vocab, mode, case_vocab=None,polar_vocab=None,mood_vocab=None,evid_vocab=None,pos_vocab=None,per_vocab=None,num_vocab=None,tense_vocab=None,aspect_vocab=None,inter_vocab=None,poss_vocab=None):
    surf_data = []; data = []; tag_data = dict(); entry_data = []
    
    tag_vocabs = dict()

    if case_vocab is None:
        case_vocab = defaultdict(lambda: len(case_vocab))
        case_vocab['<pad>'] = 0
        case_vocab['<s>'] = 1
        case_vocab['</s>'] = 2
        case_vocab['<unk>'] = 3

    if polar_vocab is None:
        polar_vocab = defaultdict(lambda: len(polar_vocab))
        polar_vocab['<pad>'] = 0
        polar_vocab['<s>'] = 1
        polar_vocab['</s>'] = 2
        polar_vocab['<unk>'] = 3

    if mood_vocab is None:
        mood_vocab = defaultdict(lambda: len(mood_vocab))
        mood_vocab['<pad>'] = 0
        mood_vocab['<s>'] = 1
        mood_vocab['</s>'] = 2
        mood_vocab['<unk>'] = 3

    if evid_vocab is None:
        evid_vocab = defaultdict(lambda: len(evid_vocab))
        evid_vocab['<pad>'] = 0
        evid_vocab['<s>'] = 1
        evid_vocab['</s>'] = 2
        evid_vocab['<unk>'] = 3

    if pos_vocab is None:
        pos_vocab = defaultdict(lambda: len(pos_vocab))
        pos_vocab['<pad>'] = 0
        pos_vocab['<s>'] = 1
        pos_vocab['</s>'] = 2
        pos_vocab['<unk>'] = 3

    if per_vocab is None:
        per_vocab = defaultdict(lambda: len(per_vocab))
        per_vocab['<pad>'] = 0
        per_vocab['<s>'] = 1
        per_vocab['</s>'] = 2
        per_vocab['<unk>'] = 3

    if num_vocab is None:
        num_vocab = defaultdict(lambda: len(num_vocab))
        num_vocab['<pad>'] = 0
        num_vocab['<s>'] = 1
        num_vocab['</s>'] = 2
        num_vocab['<unk>'] = 3

    if tense_vocab is None:
        tense_vocab = defaultdict(lambda: len(tense_vocab))
        tense_vocab['<pad>'] = 0
        tense_vocab['<s>'] = 1
        tense_vocab['</s>'] = 2
        tense_vocab['<unk>'] = 3

    if aspect_vocab is None:
        aspect_vocab = defaultdict(lambda: len(aspect_vocab))
        aspect_vocab['<pad>'] = 0
        aspect_vocab['<s>'] = 1
        aspect_vocab['</s>'] = 2
        aspect_vocab['<unk>'] = 3

    if inter_vocab is None:
        inter_vocab = defaultdict(lambda: len(inter_vocab))
        inter_vocab['<pad>'] = 0
        inter_vocab['<s>'] = 1
        inter_vocab['</s>'] = 2
        inter_vocab['<unk>'] = 3

    if poss_vocab is None:
        poss_vocab = defaultdict(lambda: len(poss_vocab))
        poss_vocab['<pad>'] = 0
        poss_vocab['<s>'] = 1
        poss_vocab['</s>'] = 2
        poss_vocab['<unk>'] = 3

    tag_vocabs['case'] = case_vocab
    tag_data['case'] = []

    tag_vocabs['polar'] = polar_vocab
    tag_data['polar'] = []

    tag_vocabs['mood'] = mood_vocab
    tag_data['mood'] = []

    tag_vocabs['evid'] = evid_vocab
    tag_data['evid'] = []

    tag_vocabs['pos'] = pos_vocab
    tag_data['pos'] = []

    tag_vocabs['per'] = per_vocab
    tag_data['per'] = []

    tag_vocabs['num'] = num_vocab
    tag_data['num'] = []

    tag_vocabs['tense'] = tense_vocab
    tag_data['tense'] = []

    tag_vocabs['aspect'] = aspect_vocab
    tag_data['aspect'] = []

    tag_vocabs['inter'] = inter_vocab
    tag_data['inter'] = []

    tag_vocabs['poss'] = poss_vocab
    tag_data['poss'] = []
    
    
    count = 0
    surfs  = []
    if 'txt' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf,tags,entries = line.strip().split('\t')
                if surf not in surfs:
                    added_tagnames = []
                    for z in tags.split(','):
                        tagname, label = z.split('=')
                        added_tagnames.append(tagname)
                        tag_data[tagname].append([tag_vocabs[tagname][label]])
                    for tname in tag_vocabs.keys():
                        if tname not in added_tagnames:
                            tag_data[tname].append([tag_vocabs[tname]["<pad>"]])
                    surf_data.append([surface_vocab[char] for char in surf])
                    surfs.append(surf)
                    entry_data.append([int(i) for i in entries.split('-')])

    print(mode,':')
    print('surf_data:',  len(surf_data))
    print('tag_data:',  len(tag_data))
    print('entry_data:',  len(entry_data))
    for surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry in zip(surf_data, tag_data['case'], tag_data['polar'],tag_data['mood'],tag_data['evid'],tag_data['pos'],tag_data['per'],tag_data['num'],tag_data['tense'],tag_data['aspect'],tag_data['inter'],tag_data['poss'], entry_data):
        data.append([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry])
    return data, tag_vocabs
    
def build_data(args, surface_vocab=None):
    # Read data and get batches...
    #surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
    trndata, tag_vocabs = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'TRN')
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    vlddata, _ = read_data(args.maxvalsize, args.valdata, surface_vocab, 'VAL',
    tag_vocabs['case'],
    tag_vocabs['polar'],
    tag_vocabs['mood'],
    tag_vocabs['evid'],
    tag_vocabs['pos'],
    tag_vocabs['per'],
    tag_vocabs['num'],
    tag_vocabs['tense'],
    tag_vocabs['aspect'],
    tag_vocabs['inter'],
    tag_vocabs['poss'])    
    args.valsize = len(vlddata)
    vld_batches, _ = get_batches(vlddata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    tstdata, _ = read_data(args.maxtstsize, args.tstdata, surface_vocab, 'TST')
    args.tstsize = len(tstdata)
    tst_batches, _ = get_batches(tstdata, surface_vocab, args.batchsize, '') 
    return (trndata, vlddata, tstdata), (trn_batches, vld_batches, tst_batches), surface_vocab, tag_vocabs



# CONFIG

args = config()
# training
args.batchsize = 128; args.epochs = 200
args.opt= 'Adam'; args.lr = 0.01
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'


# data
args.trndata  = 'trn_4x10_shuffled.txt'
args.valdata  = 'val_4x10_shuffled.txt'
args.tstdata = args.valdata

#args.surface_vocab_file = args.trndata
args.maxtrnsize = 10000000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab, tag_vocabs = build_data(args, args.vocab)

trnbatches, valbatches, tstbatches = batches


trndict = 4

import torch
import torch.nn as nn
import torch.optim as optim

class Tagmapper(nn.Module):
    def __init__(self):
        super(Tagmapper, self).__init__()
        self.linears = nn.ModuleList([])
        self.hiddens = nn.ModuleList([])

        self.emb_dim   = 100 
        self.hid_dim_1 = 500
        
        for key,keydict in tag_vocabs.items():
            print(key, len(keydict))
            self.linears.append(nn.Linear(self.emb_dim, len(keydict)))
        for linear in self.linears:
            uniform_initializer(0.1)(linear.weight)

        self.hidden_common= nn.Linear(self.emb_dim*len(tag_vocabs) + 128, self.hid_dim_1)

        if trndict ==1:
            self.entry_1= nn.Linear(self.hid_dim_1,10)
        elif trndict ==2:
            self.entry_2= nn.Linear(self.hid_dim_1,10)
        elif trndict==3:
            self.entry_3= nn.Linear(self.hid_dim_1,10)
        elif trndict==4:
            self.entry_4= nn.Linear(self.hid_dim_1,10)


vqvae = args.model
print(vqvae)
model = Tagmapper()
print(model)
#optimizer = optim.SGD(model.parameters(), lr=1.0) 
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

criterion = nn.CrossEntropyLoss()
sft = nn.Softmax(dim=2)
model.to('cuda')
for name, prm in model.named_parameters():
    print('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))

for param in model.parameters():
    uniform_initializer(0.01)(param)



numbatches = len(trnbatches); indices = list(range(numbatches))
for epc in range(300):
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    random.shuffle(indices) # this breaks continuity if there is any
    correct = 0; total = 0
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry  = trnbatches[idx] 
        fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
        _root_fhs =  vqvae.linear_root(fwd_fhs).detach()
        
        feature_vec = torch.cat((_root_fhs.detach(),
                            model.linears[0].weight[case],
                            model.linears[1].weight[polar],
                            model.linears[2].weight[mood],
                            model.linears[3].weight[evid],
                            model.linears[4].weight[pos],
                            model.linears[5].weight[per],
                            model.linears[6].weight[num],
                            model.linears[7].weight[tense],
                            model.linears[8].weight[aspect],
                            model.linears[9].weight[inter],
                            model.linears[10].weight[poss]),dim=2)
        
        # B,1, hid_dim_1
        output = model.hidden_common(feature_vec)
        output = torch.tanh(output)
     
     
        if trndict==1:
            output_1 = model.entry_1(output)
            probs_1 = sft(output_1)
            k1_loss = criterion(probs_1.squeeze(1),entry[:,:1].squeeze(1))
            loss = k1_loss 
            epoch_loss += loss.item()
            pred_1 = torch.argmax(probs_1,dim=2)
            correct += torch.sum(pred_1 == entry[:,:1]).item()
            total += pred_1.size(0) * pred_1.size(1)

        elif trndict==2:
            output_2 = model.entry_2(output)
            probs_2 = sft(output_2)
            k2_loss = criterion(probs_2.squeeze(1),entry[:,1:2].squeeze(1))
            loss = k2_loss 
            epoch_loss += loss.item()
            pred_2 = torch.argmax(probs_2,dim=2)
            correct += torch.sum(pred_2 == entry[:,1:2]).item()
            total += pred_2.size(0) * pred_2.size(1)

        elif trndict==3:
            output_3 = model.entry_3(output)
            probs_3 = sft(output_3)
            k3_loss = criterion(probs_3.squeeze(1),entry[:,2:3].squeeze(1))
            loss = k3_loss 
            epoch_loss += loss.item()
            pred_3 = torch.argmax(probs_3,dim=2)
            correct += torch.sum(pred_3 == entry[:,2:3]).item()
            total += pred_3.size(0) * pred_3.size(1)

        elif trndict==4:
            output_4 = model.entry_4(output)
            probs_4 = sft(output_4)
            k4_loss = criterion(probs_4.squeeze(1),entry[:,3:].squeeze(1))
            loss = k4_loss 
            epoch_loss += loss.item()
            pred_4 = torch.argmax(probs_4,dim=2)
            correct += torch.sum(pred_4 == entry[:,3:]).item()
            total += pred_4.size(0) * pred_4.size(1)


        loss.backward() 
        optimizer.step()
        optimizer.zero_grad() 
    






    with torch.no_grad():
        val_epoch_loss = 0;  val_correct = 0; val_total = 0
        numvalbatches = len(valbatches)
        valindices = list(range(numvalbatches))
        for i, idx in enumerate(valindices):
            # (batchsize, t)
            surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry  = valbatches[idx] 
            fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
            _root_fhs =  vqvae.linear_root(fwd_fhs).detach()

            feature_vec = torch.cat((_root_fhs,
                                model.linears[0].weight[case],
                                model.linears[1].weight[polar],
                                model.linears[2].weight[mood],
                                model.linears[3].weight[evid],
                                model.linears[4].weight[pos],
                                model.linears[5].weight[per],
                                model.linears[6].weight[num],
                                model.linears[7].weight[tense],
                                model.linears[8].weight[aspect],
                                model.linears[9].weight[inter],
                                model.linears[10].weight[poss]),dim=2)
            
            # B,1, hid_dim_1
            output = model.hidden_common(feature_vec)
            output = torch.tanh(output)
            
            if trndict==1:
                output_1 = model.entry_1(output)
                probs_1 = sft(output_1)
                k1_loss = criterion(probs_1.squeeze(1),entry[:,:1].squeeze(1))
                loss = k1_loss 
                val_epoch_loss += loss.item()
                pred_1 = torch.argmax(probs_1,dim=2)
                val_correct += torch.sum(pred_1 == entry[:,:1]).item()
                val_total += pred_1.size(0) * pred_1.size(1)

            elif trndict==2:
                output_2 = model.entry_2(output)
                probs_2 = sft(output_2)
                k2_loss = criterion(probs_2.squeeze(1),entry[:,1:2].squeeze(1))
                loss = k2_loss 
                val_epoch_loss += loss.item()
                pred_2 = torch.argmax(probs_2,dim=2)
                val_correct += torch.sum(pred_2 == entry[:,1:2]).item()
                val_total += pred_2.size(0) * pred_2.size(1)


            elif trndict==3:
                output_3 = model.entry_3(output)
                probs_3 = sft(output_3)
                k3_loss = criterion(probs_3.squeeze(1),entry[:,2:3].squeeze(1))
                loss = k3_loss 
                val_epoch_loss += loss.item()
                pred_3 = torch.argmax(probs_3,dim=2)
                val_correct += torch.sum(pred_3 == entry[:,2:3]).item()
                val_total += pred_3.size(0) * pred_3.size(1)


            elif trndict==4:
                output_4 = model.entry_4(output)
                probs_4 = sft(output_4)
                k4_loss = criterion(probs_4.squeeze(1),entry[:,3:].squeeze(1))
                loss = k4_loss 
                val_epoch_loss += loss.item()
                pred_4 = torch.argmax(probs_4,dim=2)
                val_correct += torch.sum(pred_4 == entry[:,3:]).item()
                val_total += pred_4.size(0) * pred_4.size(1)


        
    print('epoch: %d, epoch_loss: %.4f epoch_acc: %.5f' % (epc, epoch_loss/numbatches, correct/total))
    print('epoch: %d, val_epoch_loss: %.4f val_epoch_acc: %.5f' % (epc, val_epoch_loss/numvalbatches, val_correct/val_total))
    print('\n-------------------------------------------------------------')


with torch.no_grad():
    with open('/kuacc/users/mugekural/workfolder/dev/git/trmor/data/sigmorphon2016/turkish-task3-test', 'r') as reader:
        with open('dict'+str(trndict), 'w') as writer:
            for line in reader:
                tag_data  = dict()
                surf_data = []
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
                data = []
                inflected_word,tags,entries = line.strip().split('\t')
                x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(inflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
                added_tagnames = []
                for z in tags.split(','):
                    tagname, label = z.split('=')
                    added_tagnames.append(tagname)
                    tag_data[tagname].append([tag_vocabs[tagname][label]])
                for tname in tag_vocabs.keys():
                    if tname not in added_tagnames:
                        tag_data[tname].append([tag_vocabs[tname]["<pad>"]])
                surf_data.append(x)
                
                for surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss in zip(surf_data, tag_data['case'], tag_data['polar'],tag_data['mood'],tag_data['evid'],tag_data['pos'],tag_data['per'],tag_data['num'],tag_data['tense'],tag_data['aspect'],tag_data['inter'],tag_data['poss']):
                    data.append([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss])
                # (batchsize, t)
                surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss  = data[0] 
                case = torch.tensor(case).unsqueeze(0).to('cuda')
                polar = torch.tensor(polar).unsqueeze(0).to('cuda')
                mood = torch.tensor(mood).unsqueeze(0).to('cuda')
                evid = torch.tensor(evid).unsqueeze(0).to('cuda')
                pos = torch.tensor(pos).unsqueeze(0).to('cuda')
                per = torch.tensor(per).unsqueeze(0).to('cuda')
                num = torch.tensor(num).unsqueeze(0).to('cuda')
                tense = torch.tensor(tense).unsqueeze(0).to('cuda')
                aspect = torch.tensor(aspect).unsqueeze(0).to('cuda')
                inter = torch.tensor(inter).unsqueeze(0).to('cuda')
                poss = torch.tensor(poss).unsqueeze(0).to('cuda')
                fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
                _root_fhs =  vqvae.linear_root(fwd_fhs).detach()
                feature_vec = torch.cat((_root_fhs,
                                    model.linears[0].weight[case],
                                    model.linears[1].weight[polar],
                                    model.linears[2].weight[mood],
                                    model.linears[3].weight[evid],
                                    model.linears[4].weight[pos],
                                    model.linears[5].weight[per],
                                    model.linears[6].weight[num],
                                    model.linears[7].weight[tense],
                                    model.linears[8].weight[aspect],
                                    model.linears[9].weight[inter],
                                    model.linears[10].weight[poss]),dim=2)
               
                # B,1, hid_dim_1
                output = model.hidden_common(feature_vec)
                output = torch.tanh(output)

                if trndict == 1:
                    output_1 = model.entry_1(output)
                    probs_1 = sft(output_1)
                    val_epoch_loss += loss.item()
                    pred_1 = torch.argmax(probs_1,dim=2)
                    writer.write(inflected_word+'\t'+str(pred_1.item())+'\n')
                elif trndict == 2:
                    output_2 = model.entry_2(output)
                    probs_2 = sft(output_2)
                    val_epoch_loss += loss.item()
                    pred_2 = torch.argmax(probs_2,dim=2)
                    writer.write(inflected_word+'\t'+str(pred_2.item())+'\n')
                elif trndict == 3:
                    output_3 = model.entry_3(output)
                    probs_3 = sft(output_3)
                    val_epoch_loss += loss.item()
                    pred_3 = torch.argmax(probs_3,dim=2)
                    writer.write(inflected_word+'\t'+str(pred_3.item())+'\n')
                elif trndict == 4:
                    output_4 = model.entry_4(output)
                    probs_4 = sft(output_4)
                    val_epoch_loss += loss.item()
                    pred_4 = torch.argmax(probs_4,dim=2)
                    writer.write(inflected_word+'\t'+str(pred_4.item())+'\n')