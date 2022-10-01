# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
from model.vqvae.sig2016.early_sup.vqvae_kl_bi_early_sup import VQVAE
from common.utils import *
from common.vocab import VocabEntry
from torch import optim
from evaluation.probing.sig2016.probe import Probe
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
from model.msved.msved_no_att import MSVED
probe_morph_tag = 'tense'
def test(batches, args, epc):
    numbatches = len(batches); indices = list(range(numbatches))
    best_loss = 1e4
    epoch_loss = 0; epoch_acc = 0; epoch_tokens = 0
    random.shuffle(indices)
    for j, lidx in enumerate(indices):
        _, tags, lxtgt  = batches[lidx] 
        morph_tags= tags[args.tag_key_id]
        mu, logvar, encoder_fhs = args.model.encoder(lxtgt)
       
        _root_fhs = torch.permute(mu.unsqueeze(0), (1,0,2)).contiguous()
        gumbel_logits, gumbel_tag_embeddings, _, _, _, preds = args.model.classifier_loss(encoder_fhs, 0.5, hard=True)
        concat_tag = torch.cat(gumbel_tag_embeddings,dim=2)

        #batch_loss, (acc,pred_tokens) = args.probe.probe_loss(_root_fhs, morph_tags)
        batch_loss, (acc,pred_tokens) = args.probe.probe_loss(concat_tag, morph_tags)
        epoch_loss += batch_loss.sum() 
        epoch_acc  += acc
        epoch_tokens += len(pred_tokens)
    loss =  epoch_loss / epoch_tokens
    acc  =  epoch_acc / epoch_tokens  
    print('\nVAL epc %d, loss: %.3f, acc: %.3f' % (epc, loss, acc))
    #['case', 'polar', 'mood', 'evid', 'pos', 'per', 'num', 'tense', 'aspect', 'inter', 'poss']

def train(data, args):
    _, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, valbatches ,_, ubatches = data
    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.probe.parameters()), lr=args.lr,  weight_decay=1e-5)
    numbatches = len(ubatches); indices = list(range(numbatches))
    numlxtgtbatches = len(lxtgt_ordered_batches); ltgtindices = list(range(numlxtgtbatches))
    best_loss = 1e4
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_tokens = 0
        args.logger.write('\n-----------------------------------------------------\n')
        random.shuffle(ltgtindices)
        for j, lidx in enumerate(ltgtindices):
            args.probe.zero_grad()
            _, tags, lxtgt  = lxtgt_ordered_batches[lidx] 
            morph_tags= tags[args.tag_key_id]

            mu, logvar, encoder_fhs = args.model.encoder(lxtgt)
            _root_fhs = torch.permute(mu.unsqueeze(0), (1,0,2)).contiguous()
            gumbel_logits, gumbel_tag_embeddings, _, _, _, preds = args.model.classifier_loss(encoder_fhs, 0.5, hard=True)
            concat_tag = torch.cat(gumbel_tag_embeddings,dim=2)

            #batch_loss, (acc,pred_tokens) = args.probe.probe_loss(_root_fhs, morph_tags)
            batch_loss, (acc,pred_tokens) = args.probe.probe_loss(concat_tag, morph_tags)
            batch_loss.mean().backward()
            opt.step()
            epoch_loss += batch_loss.sum() 
            epoch_acc  += acc
            epoch_tokens += len(pred_tokens)
        loss =  epoch_loss / epoch_tokens
        acc  =  epoch_acc / epoch_tokens  
        print('epc %d, loss: %.3f, acc: %.3f' % (epc, loss, acc))
        test(valbatches, args, epc)
        #['case', 'polar', 'mood', 'evid', 'pos', 'per', 'num', 'tense', 'aspect', 'inter', 'poss']
            
# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 301
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'
args.kl_max = 0.2
dataset_type = 'V'
args.lang='turkish'


_model_path, surf_vocab, tag_vocabs  = get_model_info('turkish_late_msved') 
# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)
with open(tag_vocabs) as f:
    args.tag_vocabs = json.load(f)



# data
args.trndata  = 'data/sigmorphon2016/'+args.lang+'-task3-train'
args.valdata  = 'data/sigmorphon2016/'+args.lang+'-task3-test'
args.tstdata  = args.valdata
args.unlabeled_data = 'data/sigmorphon2016/'+args.lang+'_ux.txt'
args.maxtrnsize = 1000000; args.maxvalsize = 1000; args.maxtstsize = 1000000000000; args.maxusize=1000000


from model.msved.data.data_2 import build_data
rawdata, batches, _, _ = build_data(args, args.surf_vocab, args.tag_vocabs)
tag_names = [k for k in args.tag_vocabs.keys()]
args.tag_key_id = tag_names.index(probe_morph_tag)


# model
args.mname = 'msved' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.ni = 300; 
args.enc_nh = 256;
args.dec_nh = 256  
args.nz = 128; 
args.tag_embed_dim=60
args.model = MSVED(args, args.surf_vocab, args.tag_vocabs, model_init, emb_init)
args.model.load_state_dict(torch.load(_model_path))
args.model.eval()
for param in args.model.parameters():
    param.requires_grad = False
args.model.to(args.device)

args.nh = 660#args.nz
args.probe = Probe(args, len(args.tag_vocabs['tense']))
args.probe.to(args.device)

# data
args.trndata  = 'evaluation/probing/sig2016/tense/tense.trn.txt'
args.valdata  = 'evaluation/probing/sig2016/tense/tense.val.txt'
args.tstdata  = args.valdata

rawdata, batches, _, _ = build_data(args, args.surf_vocab, args.tag_vocabs)
trndata, valdata, tstdata, udata = rawdata
args.trnsize , args.valsize, args.tstsize, args.usize = len(trndata), len(valdata), len(tstdata), len(udata)

# logging
args.modelname = 'evaluation/probing/sig2016/results/'+args.lang+'/'+str(args.lxtgtsize)+ '_instances/'

try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)

args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# RUN
train(batches, args)
