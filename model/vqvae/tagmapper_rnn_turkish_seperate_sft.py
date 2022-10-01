import re, torch, json, os, argparse, random
from collections import defaultdict, Counter
from common.vocab import VocabEntry
import re, torch, json, os
from common.utils import *
from vqvae_discrete import VQVAE
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from typing import TypeVar, List

Tensor = TypeVar('torch.tensor')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
number_of_surf_tokens = 0; number_of_surf_unks = 0

## Data 
def get_batch_tagmapping(x, surface_vocab, device=device):
    global number_of_surf_tokens, number_of_surf_unks
    surf =[]; case=[]; polar =[]; mood=[]; evid=[]; pos=[]; per=[]; num=[]; tense=[]; aspect=[]; inter=[]; poss=[]; entry = [] 
    rsurf =[];
    max_surf_len = max([len(s[0]) for s in x])
    max_rsurf_len = max([len(s[-1]) for s in x])
    for surf_idx, case_idx,polar_idx, mood_idx ,evid_idx,pos_idx,per_idx,num_idx,tense_idx,aspect_idx,inter_idx,poss_idx, entry_idx, rsurf_idx  in x:
        surf_padding = [surface_vocab['<pad>']] * (max_surf_len - len(surf_idx)) 
        surf.append([surface_vocab['<s>']] + surf_idx + [surface_vocab['</s>']] + surf_padding)
        rsurf_padding = [surface_vocab['<pad>']] * (max_rsurf_len - len(rsurf_idx)) 
        rsurf.append([surface_vocab['<s>']] + rsurf_idx + [surface_vocab['</s>']] + rsurf_padding)
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
            torch.tensor(entry, dtype=torch.long, requires_grad=False, device=device),\
            torch.tensor(rsurf, dtype=torch.long, requires_grad=False, device=device)

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
        if seq_to_no_pad == 'feature':
            z = sorted(zip(order, data), key=lambda i: len(i[1][-1]))
    if seq_to_no_pad == 'surface':
        z = sorted(zip(order, data), key=lambda i: len(i[1][0]))
    #else:
    #    z = sorted(zip(order, data), key=lambda i: len(i[1][-1]))

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
                while jr < min(len(data), i+batchsize) and len(data[jr][-1]) == len(data[i][-1]): 
                    jr += 1
            batches.append(get_batch_tagmapping(data[i: jr], vocab, device=device))
            i = jr
        else:
            batches.append(get_batch_tagmapping(data[i: i+batchsize], vocab, device=device))
            i += batchsize
    return batches, order    


def read_data(maxdsize, file, surface_vocab, mode, case_vocab=None,polar_vocab=None,mood_vocab=None,evid_vocab=None,pos_vocab=None,per_vocab=None,num_vocab=None,tense_vocab=None,aspect_vocab=None,inter_vocab=None,poss_vocab=None, voice_vocab=None):
    surf_data = []; data = []; tag_data = dict(); entry_data = []
    rsurf_data = [];
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
    rsurfs  = []

    if 'txt' in file:
        with open(file, 'r') as reader:
            for line in reader:     
                count += 1
                if count > maxdsize:
                    break
                surf,tags,entries, rsurf = line.strip().split('\t')
                if True:#surf not in surfs:
                    added_tagnames = []
                    for z in tags.split(','):
                        tagname, label = z.split('=')
                        added_tagnames.append(tagname)
                        tag_data[tagname].append([tag_vocabs[tagname][label]])
                    for tname in tag_vocabs.keys():
                        if tname not in added_tagnames:
                            tag_data[tname].append([tag_vocabs[tname]["<pad>"]])
                    surf_data.append([surface_vocab[char] for char in surf])
                    rsurf_data.append([surface_vocab[char] for char in rsurf])
                    surfs.append(surf)
                    rsurfs.append(rsurf)
                    entry_data.append([int(i) for i in entries.split('-')])

    args.logger.write(mode + ':')
    args.logger.write('\nsurf_data:' +  str(len(surf_data)))
    args.logger.write('\ntag_data:'  +  str(len(tag_data)))
    args.logger.write('\nentry_data:' + str(len(entry_data)))
    args.logger.write('\nrsurf_data:' +  str(len(rsurf_data)))
 
    with open(args.logdir+mode+'_data.txt', 'w') as writer:
        for surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry, rsurf in zip(surf_data, tag_data['case'], tag_data['polar'],tag_data['mood'],tag_data['evid'],tag_data['pos'],tag_data['per'],tag_data['num'],tag_data['tense'],tag_data['aspect'],tag_data['inter'],tag_data['poss'], entry_data, rsurf_data):
            data.append([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry, rsurf])
            writer.write(str([surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, entry, rsurf])+"\n")
        
    return data, tag_vocabs
    
def build_data(args, surface_vocab):
    # Read data and get batches...
    #surface_vocab = MonoTextData(args.surface_vocab_file, label=False).vocab
    trndata, tag_vocabs = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'TRN')
    args.trnsize = len(trndata)
    trn_batches, _ = get_batches(trndata, surface_vocab, args.batchsize, args.seq_to_no_pad) 

    lxtgt_ordered_data, _ = read_data(args.maxtrnsize, args.trndata, surface_vocab, 'LXTGT',
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
    args.lxtgt_ordered_data_size = len(lxtgt_ordered_data)
    lxtgt_ordered_batches, _ = get_batches(lxtgt_ordered_data, surface_vocab, args.batchsize, 'feature') 



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

    tstdata, _ = read_data(args.maxvalsize, args.valdata, surface_vocab, 'TST',
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
    tst_batches, _ = get_batches(vlddata, surface_vocab, 1) 
    return (trndata, lxtgt_ordered_data, vlddata, tstdata), (trn_batches, lxtgt_ordered_batches, vld_batches, tst_batches), surface_vocab, tag_vocabs

## Model
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, target, latents: Tensor,epc, forceid=-1, normalize=True) -> Tensor:

        # latents: (batch_size, 1, enc_nh)
        latents = latents.contiguous()  
        latents_shape = latents.shape
        batch_size, t, emb_dim = latents.shape
        # (batch_size * t, D)
        flat_latents = latents.view(batch_size * t, self.D)  

        # Get the encoding that has the min distance
        # (batch_size * t, 1)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight** 2, dim=1) - \
                2 * torch.matmul(flat_latents, self.embedding.weight.t())
        
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  
        if target is None:
            return  encoding_inds.t()
        lossy_encoding_inds = target.unsqueeze(1)
        #encoding_inds = torch.argmax(F.cosine_similarity(flat_latents.unsqueeze(1), self.embedding.weight, dim=-1),dim=1).unsqueeze(1)
       
        #if forceid> -1:
        #    encoding_inds =  torch.LongTensor([[forceid]])

        # Convert to one-hot encodings
        encoding_one_hot = torch.zeros(lossy_encoding_inds.size(0), self.K, device=latents.device)
        # (batch_size * t, K)
        encoding_one_hot.scatter_(1, lossy_encoding_inds, 1) 
        # Quantize the latents
        # (batch_size * t, D)
        quantized_latents = torch.matmul(encoding_one_hot,  self.embedding.weight) 
        # (batch_size, t, D)
        quantized_latents = quantized_latents.view(latents_shape)  
        # Compute the VQ Losses (avg over all b*t*d)
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents, reduce=False).mean(-1)
        return commitment_loss, encoding_inds.t()

class Tagmapper(nn.Module):
    def __init__(self):
        super(Tagmapper, self).__init__()

        self.emb_dim   = args.emb_dim
        self.rnn_dim   = args.enc_nh 
        self.orddict_emb_dim = int(args.enc_nh/args.num_dicts)
        self.embeddings = nn.ModuleList([])


        for key,keydict in tag_vocabs.items():
            print(key, len(keydict))
            self.embeddings.append(nn.Embedding(len(keydict), self.emb_dim))

        if 'late' in args.model_id:
            self.head = nn.Linear(self.rnn_dim, vqvae.ord_vq_layers[args.dict_no].embedding.weight.size(0))
     

        self.encoder = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=self.rnn_dim,
                                num_layers=1,
                                batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.2)

# CONFIG
def config():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'turkish_late'
    args.lang ='turkish'

    args.model_id = model_id
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.dec_nh = 512  
    args.beta = 0.2
    args.nz = 128; 
    args.num_dicts = 11
    args.outcat=0; 
    args.orddict_emb_num =  6
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)


    if 'late' in model_id:
        from model.vqvae.sig2016.early_sup.vqvae_kl_bi_early_sup import VQVAE

        model_path, model_vocab, tag_vocabs  = get_model_info(model_id, lang=args.lang)
        with open(model_vocab) as f:
            word2id = json.load(f)
            args.surf_vocab = VocabEntry(word2id)

        args.tag_vocabs = dict()
        args.enc_nh = 660
        args.incat = args.enc_nh; 
        args.emb_dim = args.dec_nh
        with open(tag_vocabs) as f:
            args.tag_vocabs = json.load(f)
        args.model = VQVAE(args, args.surf_vocab, args.tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat')
    
  
    args.dict_no = 0 
    # logging
    args.logdir = 'model/vqvae/results/tagmapping_rnn_seperate/dict'+str(args.dict_no)+'/'+args.lang+'/'+model_id+'/'
    args.logfile = args.logdir + '/copies.txt'
    args.epochs = 200
    
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")


    args.save_path = args.logdir +  str(args.epochs)+'epochs.pt'
    args.log_path =  args.logdir +  str(args.epochs)+'epochs.log'
    args.logger = Logger(args.log_path)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    args.model.to('cuda')
    return args
args = config()

from vqvae_tag_analysis import tag_analysis, counter
#tag_analysis(args)

counter(args.model_id, args.lang, args.logger)

# training
args.batchsize = 128; 
args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'

# data
args.trndata  = 'model/vqvae/results/tagmapping_rnn/'+args.lang+'/'+args.model_id+'/train_'+args.model_id+'_shuffled.txt'
args.valdata  = 'model/vqvae/results/tagmapping_rnn/'+args.lang+'/'+args.model_id+'/test_'+args.model_id+'_shuffled.txt'
args.tstdata = args.valdata

args.maxtrnsize = 100000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, vocab, tag_vocabs = build_data(args, args.surf_vocab)
trnbatches, lxtgt_ordered_batches, valbatches, tstbatches = batches

vqvae = args.model
model = Tagmapper()
#for i in range(len(model.ord_vq_layers)):
#    model.ord_vq_layers[i].embedding.weight = vqvae.ord_vq_layers[i].embedding.weight
#    model.ord_vq_layers[i].embedding.weight.requires_grad = False

#model.encoder.embed = vqvae.encoder.embed
#model.encoder.lstm  = vqvae.encoder.lstm
model.to('cuda')
args.logger.write(vqvae)
args.logger.write(model)
for name, prm in model.named_parameters():
    args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
xloss = nn.CrossEntropyLoss()

def train(args, lxsrc_ordered_batches, lxtgt_ordered_batches, valbatches, model, vqvae, optimizer, epc):
    numlxsrc_batches = len(lxsrc_ordered_batches); lxsrc_indices = list(range(numlxsrc_batches))
    numlxtgt_batches = len(lxtgt_ordered_batches); lxtgt_indices = list(range(numlxtgt_batches))

    args.logger.write('\n------------------------------------------')
    model.train()
    epoch_loss = 0; correct = 0; total = 0
    random.shuffle(lxsrc_indices)
    random.shuffle(lxtgt_indices)

    
    for _, idx in enumerate(lxsrc_indices):
        ## surf-> rsurf
        # (batchsize, t)
        surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss,entry, rsurf= lxsrc_ordered_batches[idx] 
        _, _, _, mu, logvar, fwd,bck = vqvae.encoder(surf)
        _root_fhs = mu.unsqueeze(1)
        _root_fhs = vqvae.z_to_dec(_root_fhs)
        feature_vec = torch.cat((_root_fhs.detach(),
                            model.embeddings[0].weight[case],
                            model.embeddings[1].weight[polar],
                            model.embeddings[2].weight[mood],
                            model.embeddings[3].weight[evid],
                            model.embeddings[4].weight[pos],
                            model.embeddings[5].weight[per],
                            model.embeddings[6].weight[num],
                            model.embeddings[7].weight[tense],
                            model.embeddings[8].weight[aspect],
                            model.embeddings[9].weight[inter],
                            model.embeddings[10].weight[poss]),dim=1)
        feature_vec = model.dropout(feature_vec)
        _, (last_state, _) = model.encoder(feature_vec)
        
        sft = nn.Softmax(dim=1)
        logits =  model.head(last_state.squeeze(0))
        loss = xloss(logits, entry[:,args.dict_no])
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad() 
        preds = torch.argmax(sft(logits),dim=1)
        correct += torch.sum(preds == entry[:,args.dict_no]).item()
        total += len(preds)
        epoch_loss += loss.item()

    args.logger.write('\nepoch: %d, epoch_loss: %.4f epoch_acc: %.5f' % (epc, epoch_loss, correct/total))
    val_epoch_loss = 0
    exact_acc = 0
  
    with torch.no_grad():
        val_epoch_loss, acc, exact_acc= test(args, valbatches, model, vqvae, epc)
    return val_epoch_loss, exact_acc

def test(args, batches, model, vqvae, epc, log_shared_task=False):
    numbatches = len(batches); indices = list(range(numbatches))
    epoch_loss = 0; correct = 0; total = 0
    #random.shuffle(indices) # this breaks continuity if there is any
    true = 0
    model.eval()
    _preds = []

    for _, idx in enumerate(indices):
        # (batchsize, t)
        surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss,entry, rsurf = batches[idx] 
        fhs, _, _, mu, logvar, fwd,bck = vqvae.encoder(surf)
        _root_fhs = mu.unsqueeze(1)
        _root_fhs = vqvae.z_to_dec(_root_fhs)
        feature_vec = torch.cat((_root_fhs.detach(),
                            model.embeddings[0].weight[case],
                            model.embeddings[1].weight[polar],
                            model.embeddings[2].weight[mood],
                            model.embeddings[3].weight[evid],
                            model.embeddings[4].weight[pos],
                            model.embeddings[5].weight[per],
                            model.embeddings[6].weight[num],
                            model.embeddings[7].weight[tense],
                            model.embeddings[8].weight[aspect],
                            model.embeddings[9].weight[inter],
                            model.embeddings[10].weight[poss]),dim=1)

        _, (last_state, _) = model.encoder(feature_vec)
        sft = nn.Softmax(dim=1)
        logits =  model.head(last_state.squeeze(0))
        loss = xloss(logits, entry[:,args.dict_no])
        preds = torch.argmax(sft(logits),dim=1)
        correct += torch.sum(preds == entry[:,args.dict_no]).item()
        total += len(preds)
        epoch_loss += loss.item()

        if log_shared_task:
            _preds.append(''.join(args.surf_vocab.decode_sentence(surf[0]))+'\t'+ str(preds.item()))    
    args.logger.write('\nVAL epoch: %d, epoch_loss: %.4f epoch_acc: %.5f' % (epc, epoch_loss, correct/total))
    return epoch_loss, (correct/total), _preds


try:
    best_acc = 0; best_loss= 1e4
    reinflect_accs = []
    for epc in range(args.epochs):
        val_epoch_loss, val_exact_acc = train(args, trnbatches, lxtgt_ordered_batches, valbatches, model, vqvae, optimizer,epc)
        if val_epoch_loss < best_loss:
            args.logger.write('\nupdate best loss \n')
            best_loss = val_epoch_loss    
    
        if epc%3==0 or epc>100:
            with torch.no_grad():
                _, acc, preds = test(args, tstbatches, model, vqvae, epc, log_shared_task=True)
                if acc > best_acc:
                    best_acc = acc
                    with open(args.logdir+str(args.dict_no)+'_preds.txt', 'w') as predwriter:
                        for p in preds:
                            predwriter.write(p+'\n')
                reinflect_accs.append(acc)
        #scheduler.step(val_epoch_loss)
    args.logger.write('\nBest reinflection acc: %.4f' % max(reinflect_accs))
except KeyboardInterrupt:
    args.logger.write('\nBest reinflection acc: %.4f' % max(reinflect_accs))
