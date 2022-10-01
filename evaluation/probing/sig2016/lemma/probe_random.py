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
probe_part = 'quantized'

def test(batches, args, epc):
    numbatches = len(batches); indices = list(range(numbatches))
    best_loss = 1e4
    epoch_loss = 0; epoch_acc = 0; epoch_tokens = 0
    args.logger.write('\n-----------------------------------------------------\n')
    random.shuffle(indices)
    for j, lidx in enumerate(indices):
        surf, tags, lemma  = batches[lidx] 
        batch_loss, (acc,pred_tokens) = args.probe.probe_random_loss(lemma)
        epoch_loss += batch_loss
        epoch_acc  += acc
        epoch_tokens += len(pred_tokens)
    loss =  epoch_loss / epoch_tokens
    acc  =  epoch_acc / epoch_tokens  
    print('VAL epc %d, loss: %.3f, acc: %.3f' % (epc, loss, acc))
    #['case', 'polar', 'mood', 'evid', 'pos', 'per', 'num', 'tense', 'aspect', 'inter', 'poss']

def train(data, args):
    _, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, valbatches ,_, ubatches = data
    test(valbatches, args, 0) 


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


_model_path, surf_vocab, tag_vocabs  = get_model_info('turkish_late') 
# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)
with open(tag_vocabs) as f:
    args.tag_vocabs = json.load(f)


# data
args.trndata  = 'data/sigmorphon2016/'+args.lang+'-task1-train'
args.valdata  = 'data/sigmorphon2016/'+args.lang+'-task1-dev'
args.tstdata  = args.valdata
args.unlabeled_data = 'data/sigmorphon2016/'+args.lang+'_ux.txt'
args.maxtrnsize = 1000000; args.maxvalsize = 1000; args.maxtstsize = 1000000000000; args.maxusize=1000000



if args.lang == 'turkish':
    from evaluation.probing.sig2016.lemma.lemma_data import build_data



rawdata, batches, _, tag_vocabs, lemma_vocab = build_data(args, args.surf_vocab)
args.id2lemma = dict() 
for key, val in lemma_vocab.items():
    args.id2lemma[val] = key


trndata, valdata, tstdata, udata = rawdata
args.trnsize , args.valsize, args.tstsize, args.usize = len(trndata), len(valdata), len(tstdata), len(udata)

# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.ni = 256; 
if args.lang=='georgian':
    args.enc_nh = 880;
elif args.lang =='russian':
    args.enc_nh = 1320
else:
    args.enc_nh = 660;
args.dec_nh = 512  
args.embedding_dim = args.enc_nh
args.beta = 0.1
args.nz = 128; 
args.num_dicts = len(args.tag_vocabs)
args.outcat=0; 
args.orddict_emb_num = 1
args.incat = args.enc_nh; 

args.num_dicts_tmp = args.num_dicts; args.outcat_tmp=args.outcat; args.incat_tmp = args.incat; args.dec_nh_tmp = args.dec_nh
args.model = VQVAE(args, args.surf_vocab,  args.tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)
#args.model.load_state_dict(torch.load(_model_path))
args.model.eval()
for param in args.model.parameters():
    param.requires_grad = False
args.model.to(args.device)

args.nh = 660#args.nz
if probe_part == 'quantized':
    args.nh = args.enc_nh
args.probe = Probe(args, len(lemma_vocab))
args.probe.to(args.device)

# logging
args.modelname = 'evaluation/probing/sig2016/lemma/results/'+args.lang+'/'+str(args.lxtgtsize)+ '_instances/' 

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

