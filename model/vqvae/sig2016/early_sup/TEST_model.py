# -----------------------------------------------------------
# Date:        2022/02/14 <3 
# Author:      Muge Kural
# Description: Trainer of character-based VQ-variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os
from turtle import update
from unicodedata import bidirectional
import matplotlib.pyplot as plt
from model.vqvae.sig2016.early_sup.vqvae_kl_bi_early_sup import VQVAE
from model.vqvae.vqvae_ae import VQVAE_AE

from model.ae.ae import AE
from common.utils import *
from common.vocab import VocabEntry
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import defaultdict, OrderedDict
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   


def shared_task_gen_direct(args, tstbatches,epc):
    i=0
    true=0
    numbatches = len(tstbatches); indices = list(range(numbatches))
    reinflects = []
    true_ones=[]; false_ones=[]

    for _, idx in enumerate(indices):
        i+=1
        
        # (batchsize, t)
        surf,tags,rsurf = tstbatches[idx] 
        reinflect_tag = [t.item() for t in tags]
        fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(surf)
        root_z = mu.unsqueeze(1)
        root_z = args.model.z_to_dec(root_z)
        bosid = args.surf_vocab.word2id['<s>']
        input = torch.tensor(bosid).to('cuda')
        sft = nn.Softmax(dim=1)
        # Quantized Inputs
        vq_vectors = []
        vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=2:
            vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=4:
            vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
            vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=5:
            vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=6:
            vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=8:
            vq_vectors.append(args.model.ord_vq_layers[6].embedding.weight[reinflect_tag[6]].unsqueeze(0).unsqueeze(0))
            vq_vectors.append(args.model.ord_vq_layers[7].embedding.weight[reinflect_tag[7]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=10:
            vq_vectors.append(args.model.ord_vq_layers[8].embedding.weight[reinflect_tag[8]].unsqueeze(0).unsqueeze(0))
            vq_vectors.append(args.model.ord_vq_layers[9].embedding.weight[reinflect_tag[9]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=11:
            vq_vectors.append(args.model.ord_vq_layers[10].embedding.weight[reinflect_tag[10]].unsqueeze(0).unsqueeze(0))
        if args.num_dicts >=12:
            vq_vectors.append(args.model.ord_vq_layers[11].embedding.weight[reinflect_tag[11]].unsqueeze(0).unsqueeze(0))
        
        suffix_z = torch.cat(vq_vectors,dim=2)
        batch_size, seq_len, _ = fhs.size()
        z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)
        c_init = root_z
        h_init = torch.tanh(c_init)
        decoder_hidden = (h_init, c_init)
        MAX_LENGTH = 50
        pred_word = []
        c=0
        pred_form = ''
        while c<MAX_LENGTH:
            c+=1
            # (1,1,ni)
            word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0).to('cuda'))
            word_embed = torch.cat((word_embed, z_), -1)
            # output: (1,1,dec_nh)
            output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
            # (1, vocab_size)
            output_logits = args.model.decoder.pred_linear(output).squeeze(1)
            input = torch.argmax(sft(output_logits))
            char = args.surf_vocab.id2word(input.item())
            pred_word.append(char)
            if char == '</s>':
                pred_form =''.join(pred_word)
                break
        gold_form = ''.join(args.surf_vocab.decode_sentence(rsurf.squeeze(0)[1:]))
        if gold_form == pred_form:
            true +=1    
            true_ones.append(gold_form + '\t'+pred_form+'\n')
        else:
            false_ones.append(gold_form + '\t'+pred_form+'\n')

        reinflects.append(gold_form+'\t'+pred_form+'\n')
    acc = (true/i)
    args.logger.write('\nTST reinf acc direct: %.3f' % acc)
    return acc, (true_ones,false_ones)



# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 301
args.opt= 'Adam'; args.lr = 0.001
args.task = 'vqvae'
args.seq_to_no_pad = 'surface'
args.lang='finnish'
args.run_id = 1
args.beta = 0.1


if args.lang == 'georgian':
    args.dec_nh =  256  
else:
    args.dec_nh = 512  

if args.lang=='georgian':
    args.enc_nh = 640;
elif args.lang =='russian' or args.lang == 'finnish':
    args.enc_nh = 1320
else:
    args.enc_nh = 660;


_model_id = args.lang+'_thesis_sig2016'
_model_path, surf_vocab  = get_model_info(_model_id) 
# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(surf_vocab) as f:
    word2id = json.load(f)
    args.surf_vocab = VocabEntry(word2id)

# data
args.trndata  = 'data/sigmorphon2016/'+args.lang+'-task3-train'
args.valdata  = 'data/sigmorphon2016/'+args.lang+'-task3-test'
args.tstdata  = args.valdata
args.unlabeled_data = 'data/sigmorphon2016/'+args.lang+'_ux.txt'
args.maxtrnsize = 1000000; args.maxvalsize = 1000; args.maxtstsize = 1000000000000; args.maxusize=1000000
args.ux_weight = 0
args.ux_start_epc = 0

if args.lang == 'finnish':
    from  model.vqvae.data.data_2_finnish import build_data
elif args.lang == 'turkish':
    from data.data_2_turkish import build_data
elif args.lang == 'hungarian':
    from model.vqvae.data.data_2_hungarian import build_data
elif args.lang == 'maltese':
    from model.vqvae.data.data_2_maltese import build_data
elif args.lang == 'navajo':
    from data.data_2_navajo import build_data
elif args.lang == 'russian':
    from model.vqvae.data.data_2_russian import build_data
elif args.lang == 'arabic':
    from data.data_2_arabic import build_data
elif args.lang == 'german':
    from model.vqvae.data.data_2_german import build_data
elif args.lang == 'spanish':
    from  model.vqvae.data.data_2_spanish import build_data
elif args.lang == 'georgian':
    from model.vqvae.data.data_2_georgian import build_data


rawdata, batches, _, tag_vocabs = build_data(args, args.surf_vocab)
trndata, valdata, tstdata, udata = rawdata
args.trnsize , args.valsize, args.tstsize, args.usize = len(trndata), len(valdata), len(tstdata), len(udata)
# model
args.mname = 'vqvae' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
args.ni = 256; 

args.embedding_dim = args.enc_nh
args.nz = 128; 
args.num_dicts = len(tag_vocabs)
args.outcat=0; 
args.orddict_emb_num = 1
args.incat = args.enc_nh; 
args.load_from_pretrained = False
args.model = VQVAE(args, args.surf_vocab,  tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)
args.model.load_state_dict(torch.load(_model_path))

args.model.eval()
args.model.to(args.device)


# tensorboard
# load pretrained ae weights
args.model_prefix = 'run'+str(args.run_id)+'-batchsize'+str(args.batchsize)+'_beta'+str(args.beta)+'_'+str(args.num_dicts)+"x"+str(args.orddict_emb_num)+'_dec'+str(args.dec_nh)+'_suffixd'+str(args.incat)+'/'
writer = SummaryWriter("runs/early-supervision-thesis-TEST/"+args.lang+'/'+ args.model_prefix)




# logging
args.modelname = 'model/'+args.mname+'/results/training/sig2016/'+args.lang+'/early-supervision-thesis-TEST/'+str(args.lxtgtsize)+'_instances/'+args.model_prefix

try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)

with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(args.surf_vocab.word2id))

j=0
for key,val in tag_vocabs.items():
    with open(args.modelname+'/'+str(j)+'_tagvocab.json', 'w') as f:
        f.write(json.dumps(val))
        j+=1

args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')

# RUN
_, lxtgt_ordered_batches, lxtgt_ordered_batches_TST, valbatches ,_, ubatches = batches
shared_acc_direct,(true_ones_directs, false_ones_directs) = shared_task_gen_direct(args,lxtgt_ordered_batches_TST,0)
with open( args.modelname+'_SIGMORPHON2016_TRUE_DIRECTS.txt', 'w') as writer_true_directs:
    with open( args.modelname+'_SIGMORPHON2016_FALSE_DIRECTS.txt', 'w') as writer_false_directs:
        for true in true_ones_directs:
            writer_true_directs.write(true)
        for false in false_ones_directs:
            writer_false_directs.write(false)
writer.close()

