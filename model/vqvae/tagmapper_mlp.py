import re, torch, json, os, argparse, random
from collections import defaultdict, Counter
from common.vocab import VocabEntry
import re, torch, json, os
from common.utils import *
from vqvae import VQVAE
from vqvae_discrete import VQVAE
from vqvae_kl import VQVAE
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
number_of_surf_tokens = 0; number_of_surf_unks = 0

## Data prep
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
    args.logger.write('seq not to pad: %s, continuity: %s' % (seq_to_no_pad,continuity))
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
    #args.logger.write('# of surf tokens: ' + number_of_surf_tokens, ', # of surf unks: ', number_of_surf_unks)
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

    if polar_vocab is None:
        polar_vocab = defaultdict(lambda: len(polar_vocab))
        polar_vocab['<pad>'] = 0

    if mood_vocab is None:
        mood_vocab = defaultdict(lambda: len(mood_vocab))
        mood_vocab['<pad>'] = 0
    
    if evid_vocab is None:
        evid_vocab = defaultdict(lambda: len(evid_vocab))
        evid_vocab['<pad>'] = 0

    if pos_vocab is None:
        pos_vocab = defaultdict(lambda: len(pos_vocab))
        pos_vocab['<pad>'] = 0

    if per_vocab is None:
        per_vocab = defaultdict(lambda: len(per_vocab))
        per_vocab['<pad>'] = 0

    if num_vocab is None:
        num_vocab = defaultdict(lambda: len(num_vocab))
        num_vocab['<pad>'] = 0

    if tense_vocab is None:
        tense_vocab = defaultdict(lambda: len(tense_vocab))
        tense_vocab['<pad>'] = 0

    if aspect_vocab is None:
        aspect_vocab = defaultdict(lambda: len(aspect_vocab))
        aspect_vocab['<pad>'] = 0

    if inter_vocab is None:
        inter_vocab = defaultdict(lambda: len(inter_vocab))
        inter_vocab['<pad>'] = 0
 
    if poss_vocab is None:
        poss_vocab = defaultdict(lambda: len(poss_vocab))
        poss_vocab['<pad>'] = 0
 

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

    args.logger.write(mode + ':')
    args.logger.write('\nsurf_data:' +  str(len(surf_data)))
    args.logger.write('\ntag_data:'  +  str(len(tag_data)))
    args.logger.write('\nentry_data:' + str(len(entry_data)))
    with open(args.logdir+mode+'_data.txt', 'w') as writer:
        for surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry in zip(surf_data, tag_data['case'], tag_data['polar'],tag_data['mood'],tag_data['evid'],tag_data['pos'],tag_data['per'],tag_data['num'],tag_data['tense'],tag_data['aspect'],tag_data['inter'],tag_data['poss'], entry_data):
            data.append([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry])
            writer.write(str([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry])+"\n")
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

## Model
class Tagmapper(nn.Module):
    def __init__(self):
        super(Tagmapper, self).__init__()
        self.linears = nn.ModuleList([])
        self.entries = nn.ModuleList([])

        self.emb_dim   = 100 
        self.hid_dim_1 = 500
        self.hid_dim_2 = 350
        self.hid_dim_3 = 200

        for key,keydict in tag_vocabs.items():
            args.logger.write(key +':'+ str(len(keydict)))
            self.linears.append(nn.Embedding(len(keydict), self.emb_dim))

        for linear in self.linears:
            uniform_initializer(0.1)(linear.weight)

        self.hidden_common= nn.Linear(self.emb_dim*len(tag_vocabs) + args.dec_nh, self.hid_dim_1)
        self.hidden_2 = nn.Linear(self.hid_dim_1, self.hid_dim_2)
        self.hidden_3 = nn.Linear(self.hid_dim_2, self.hid_dim_3)

        self.dropout = nn.Dropout(p=0.1)

        #for i in range(1):#(args.num_dicts):
        for i in range(args.num_dicts):
            self.entries.append(nn.Linear(self.hid_dim_3,6))


def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.1_8x6_dec128_suffixd512'
    model_id = 'bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'discretelemma_bilstm_8x6_dec512_suffixd512'
    model_id = 'bi_kl0.1_16x6_dec128_suffixd512'

    args.model_id = model_id
    args.num_dicts = 16
    args.orddict_emb_num  = 6
    args.lemmadict_emb_num  = 5000
    args.nz = 128; 
    args.root_linear_h = args.nz
    args.enc_nh = 512;
    args.dec_nh = args.root_linear_h
    args.incat = args.enc_nh

    if 'uni' in model_id:
        from vqvae import VQVAE
        args.model_type = 'unilstm'

    if 'kl' in model_id:
        from vqvae_kl import VQVAE
        args.model_type = 'kl'
        args.dec_nh = args.nz  

    if 'bi' in model_id:
        from vqvae_bidirect import VQVAE
        args.model_type = 'bilstm'

    if 'discrete' in model_id:
        from vqvae_discrete import VQVAE
        args.num_dicts = 9
        args.dec_nh = args.enc_nh
        args.model_type = 'discrete'
    
    if 'bi_kl' in model_id:
        from vqvae_kl_bi import VQVAE
        args.model_type = 'bi_kl'
        args.dec_nh = args.nz  

    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/tagmapping/'+model_id+'/'
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
    args.epochs = 2000
    args.lr = 0.001
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.ni = 256; 
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.embedding_dim = args.enc_nh; 
    args.beta = 0.5
    args.outcat=0; 
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

    if args.model_type == 'discrete':
        args.num_dicts -= 1 

    args.save_path = args.logdir +  str(args.epochs)+'epochs.pt'
    args.log_path =  args.logdir +  str(args.epochs)+'epochs.log'
    args.logger = Logger(args.log_path)

    args.num_dicts = 1
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    args.model.to('cuda')
    return args

# CONFIG
args = config()
# training
args.batchsize = 128; 
args.opt= 'Adam'; 
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'

# data
args.trndata  = 'model/vqvae/results/analysis/'+args.model_id+'/train_'+args.model_id+'_shuffled.txt'
args.valdata  = 'model/vqvae/results/analysis/'+args.model_id+'/test_'+args.model_id+'_shuffled.txt'
args.tstdata = args.valdata

args.maxtrnsize = 100000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab, tag_vocabs = build_data(args, args.vocab)
trnbatches, valbatches, tstbatches = batches

dump_threshold = 0.90
dict_to_dump = 12
offset = dict_to_dump-1
writers = []
for i in range(args.num_dicts):
        writers.append(open(args.logdir+"dict"+str(i+offset)+".txt", "w"))

vqvae = args.model
model = Tagmapper()

args.logger.write(vqvae)
args.logger.write(model)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.CrossEntropyLoss()
sft = nn.Softmax(dim=2)
model.to('cuda')
model.train()
for name, prm in model.named_parameters():
    args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))

for param in model.parameters():
    uniform_initializer(0.01)(param)
for linear in model.linears:
    uniform_initializer(0.1)(linear.weight)
    
from pathlib import Path
Path(args.logdir+"preds").mkdir(parents=True, exist_ok=True)

best_loss= 9999999999999999999999
numbatches = len(trnbatches); indices = list(range(numbatches))
for epc in range(args.epochs):
    model.train()
    epoch_loss = 0; epoch_num_tokens = 0; epoch_acc = 0
    random.shuffle(indices) # this breaks continuity if there is any
    correct = 0; total = 0

    #trn
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry  = trnbatches[idx] 

        if args.model_type == 'discrete':
            fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
            _root_fhs, vq_loss, quantized_inds = vqvae.vq_layer_lemma(fwd_fhs,epc)
        
        if args.model_type == 'kl':
            fhs, _, _, mu, logvar = vqvae.encoder(surf)
            _root_fhs = mu.unsqueeze(1)
    
        if args.model_type == 'bi_kl':
            fhs, _, _, mu, logvar, fwd,bck = vqvae.encoder(surf)
            _root_fhs = mu.unsqueeze(1)

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
        output = model.dropout(output)

        output = model.hidden_2(output)
        output = torch.tanh(output)
        output = model.dropout(output)

        output = model.hidden_3(output)
        output = torch.tanh(output)
        #output = model.dropout(output)

        loss = torch.tensor(0.0).to('cuda')
        for j in range(len(model.entries)):
            logits = model.entries[j](output)
            loss += criterion(logits.squeeze(1),entry[:,j+offset:offset+j+1].squeeze(1))
            probs = sft(logits)
            pred = torch.argmax(probs,dim=2)
            correct += torch.sum(pred == entry[:,j+offset:offset+j+1]).item()
            total += pred.size(0) * pred.size(1)
        epoch_loss += loss.item()
        '''
        if epc %1000==0:
            with open(args.logdir+'preds/preds_true'+str(epc)+'.txt', 'a') as writer_true:
                with open(args.logdir+'preds/preds_false'+str(epc)+'.txt', 'a') as writer_false:
                    for i in range(entry[:,:1].shape[0]):
                        if entry[:,:1][i] != pred_1[i]:
                            writer_false.write(''.join(vocab.decode_sentence(surf[i]))+'---- gold: '+str(entry[:,:1][i].tolist())+' ---- pred:'+str(pred_1[i].tolist())+'\n')
                        else:
                            writer_true.write(''.join(vocab.decode_sentence(surf[i]))+'---- gold: '+str(entry[:,:1][i].tolist())+' ---- pred:'+str(pred_1[i].tolist())+'\n')
        preds = torch.cat((pred_1,pred_2,pred_3, pred_4, pred_5, pred_6, pred_7, pred_8),dim=1)
        if epc %100==0:
            with open(args.logdir+'preds/preds_true'+str(epc)+'.txt', 'a') as writer_true:
                with open(args.logdir+'preds/preds_false'+str(epc)+'.txt', 'a') as writer_false:
                    for i in range(entry[:,:1].shape[0]):
                        #if entry[:,:1][i] != pred_1[i]:
                        if sum(entry[i]== preds[i]) != args.num_dicts:
                            writer_false.write(''.join(vocab.decode_sentence(surf[i]))+'---- gold: '+str(entry[i].tolist())+' ---- pred:'+str(preds[i].tolist())+'\n')
                        else:
                            writer_true.write(''.join(vocab.decode_sentence(surf[i]))+'---- gold: '+str(entry[i].tolist())+' ---- pred:'+str(preds[i].tolist())+'\n')
        '''
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad() 
    args.logger.write('\nepoch: %d, epoch_loss: %.4f epoch_acc: %.5f' % (epc, epoch_loss/numbatches, correct/total))
    
    #val
    model.eval()
    with torch.no_grad():
        val_epoch_loss = 0;  val_correct = 0; val_total = 0
        numvalbatches = len(valbatches)
        valindices = list(range(numvalbatches))
        for i, idx in enumerate(valindices):
            # (batchsize, t)
            surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry  = valbatches[idx] 
            
            if args.model_type == 'discrete':
                fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
                _root_fhs, vq_loss, quantized_inds = vqvae.vq_layer_lemma(fwd_fhs,epc)
        
            if args.model_type == 'kl':
                fhs, _, _, mu, logvar = vqvae.encoder(surf)
                _root_fhs = mu.unsqueeze(1)
        
            if args.model_type == 'bi_kl':
                fhs, _, _, mu, logvar, fwd,bck = vqvae.encoder(surf)
                _root_fhs = mu.unsqueeze(1)

           
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
            
            output = model.hidden_common(feature_vec)
            output = torch.tanh(output)

            output = model.hidden_2(output)
            output = torch.tanh(output)

            output = model.hidden_3(output)
            output = torch.tanh(output)

            val_loss = torch.tensor(0.0).to('cuda')
            for j in range(len(model.entries)):
                logits = model.entries[j](output)
                val_loss += criterion(logits.squeeze(1),entry[:,offset+j:offset+j+1].squeeze(1))
                probs = sft(logits)
                pred = torch.argmax(probs,dim=2)
                val_correct += torch.sum(pred == entry[:,offset+j:offset+j+1]).item()
                val_total += pred.size(0) * pred.size(1)
            val_epoch_loss += val_loss.item()
    
    args.logger.write('\nepoch: %d, val_epoch_loss: %.4f val_epoch_acc: %.5f' % (epc, val_epoch_loss/numvalbatches, val_correct/val_total))
    if val_epoch_loss < best_loss:
        args.logger.write('\nupdate best loss \n')
        best_loss = val_epoch_loss    
        torch.save(model.state_dict(), args.save_path)
    args.logger.write('\n-------------------------------------------------------------')


    #test
    if (val_correct/val_total)< dump_threshold:
        continue
    with torch.no_grad():
        with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
            for line in reader:
                surf_data = []
                tag_data  = dict()
                tag_data['case']   = []
                tag_data['polar']  = []
                tag_data['mood']   = []
                tag_data['evid']   = []
                tag_data['pos']    = []
                tag_data['per']    = []
                tag_data['num']    = []
                tag_data['tense']  = []
                tag_data['aspect'] = []
                tag_data['inter']  = []
                tag_data['poss']   = []
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
                
                if args.model_type == 'discrete':
                    fhs, _, _, fwd_fhs, bck_fhs = vqvae.encoder(surf)
                    _root_fhs, vq_loss, quantized_inds = vqvae.vq_layer_lemma(fwd_fhs,epc)
            
                if args.model_type == 'kl':
                    fhs, _, _, mu, logvar = vqvae.encoder(surf)
                    _root_fhs = mu.unsqueeze(1)
            
                if args.model_type == 'bi_kl':
                    fhs, _, _, mu, logvar, fwd,bck = vqvae.encoder(x)
                    _root_fhs = mu.unsqueeze(1)
                
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
                
                output = model.hidden_2(output)
                output = torch.tanh(output)
                
                output = model.hidden_3(output)
                output = torch.tanh(output)

                for j in range(len(model.entries)):
                    logits = model.entries[j](output)
                    probs = sft(logits)
                    pred = torch.argmax(probs,dim=2)
                    writers[j].write(inflected_word+'\t'+str(pred.item())+'\n')
        exit()