import argparse
import torch
import torch.nn as nn
from data import MonoTextData, read_trndata_makevocab, read_valdata, get_batches
from modules import VAE, LSTMEncoder, LSTMDecoder

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)

def predict_tst_data(task, modelname):
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.task = task
 
    # LOAD MODEL & PREDICT
    # CONFIG
    args.device = 'cuda'
    args.ni = 512
    args.enc_nh = 1024
    args.dec_nh = 1024
    args.nz = 32
    args.enc_dropout_in = 0.0
    args.dec_dropout_in = 0.0
    args.dec_dropout_out = 0.0
    end = modelname.find("k")
    args.trnsize = int(modelname[0:end])*1000
    args.trndata = 'trmor_data/trmor2018.trn'
    args.valdata = 'trmor_data/trmor2018.val'
    args.tstdata = 'trmor_data/trmor2018.tst'
    ## TODO: Fix the reconstruction of source vocab!
    surface_vocab = MonoTextData(args.trndata, label=False).vocab
    trndata, feature_vocab, _ = read_trndata_makevocab(args.trndata, args.trnsize, surface_vocab) # data len:50000
    vlddata = read_valdata(args.valdata, feature_vocab, surface_vocab)     # 5000
    tstdata = read_valdata(args.tstdata, feature_vocab, surface_vocab)     # 2769
    tst_batches, _ = get_batches(tstdata, surface_vocab, feature_vocab) 
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    encoder = LSTMEncoder(args, len(feature_vocab), model_init, emb_init) 
    decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init) 
    model = VAE(encoder, decoder, args)
    model.encoder.mode = 's2s'
    model.load_state_dict(torch.load('models/'+args.task+'/'+modelname))
    print('Model weights loaded from ... ', 'models/'+args.task+'/'+modelname)
    model.eval()   
    model.to(args.device)

    '''
    with open('surf.txt', "w") as fout:
        for (surf, feat) in tst_batches:
            # surf: (Tsurf,B), feat: (Tfeat,B)
            Tsurf, B = surf.size()
            surfaces = [[] for _ in range(B)]
            for b in range(B):
                for t in range(Tsurf):
                    surfaces[b].append(surface_vocab.id2word(surf[t][b].item()))
            for sent in surfaces:
                fout.write("".join(sent[1:-1]) + '\n')
    with open('feat.txt', "w") as fout:
        for (surf, feat) in tst_batches:
            # surf: (Tsurf,B), feat: (Tfeat,B)
            Tfeat, B = feat.size()
            feats = [[] for _ in range(B)]
            for b in range(B):
                for t in range(Tfeat):
                    feats[b].append(feature_vocab.id2word(feat[t][b].item()))
            for sent in feats:
                fout.write(" ".join(sent[1:-1]) + '\n')        
    '''

    if args.task == 'feat2surf':
        with open('pred_tst_surf_'+modelname+'.txt', "w") as fout:
            for (surf, feat) in tst_batches:
                # surf: (Tsurf,B), feat: (Tfeat,B)
                decoded_batch = model.reconstruct(feat.t())
                for sent in decoded_batch[0]:
                    fout.write("".join(sent[:-1]) + "\n")

    elif args.task == 'surf2feat':
        with open('pred_tst_feat_'+modelname+'.txt', "w") as fout:
            for (surf, feat) in tst_batches:
                # surf: (Tsurf,B), feat: (Tfeat,B)
                decoded_batch = model.reconstruct(surf.t())
                for sent in decoded_batch[0]:
                    fout.write("".join(sent[:-1]) + "\n")

def predict(task, modelname, str):
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.task = task
    # LOAD MODEL & PREDICT
    # CONFIG
    args.device = 'cuda'
    args.ni = 512
    args.enc_nh = 1024
    args.dec_nh = 1024
    args.nz = 32
    args.enc_dropout_in = 0.5
    args.dec_dropout_in = 0.0
    args.dec_dropout_out = 0.0
    end = modelname.find("k")
    args.trnsize = int(modelname[0:end])*1000
    args.trndata = 'trmor_data/trmor2018.trn'
    ## TODO: Fix the reconstruction of source vocab!
    surface_vocab = MonoTextData(args.trndata, label=False).vocab
    _, feature_vocab, _ = read_trndata_makevocab(args.trndata, args.trnsize, surface_vocab) # data len:50000
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    encoder = LSTMEncoder(args, len(feature_vocab), model_init, emb_init) 
    decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init) 
    model = VAE(encoder, decoder, args)
    model.encoder.mode = 's2s'
    model.load_state_dict(torch.load('models/'+args.task+'/'+modelname))
    print('Model weights loaded from ... ', 'models/'+args.task+'/'+modelname)
    model.eval()   
    model.to(args.device)
    if task == 'feat2surf':
        feat = []
        root_and_tags = [char for char in str.replace('^','+').split('+')[0]] + str.replace('^','+').split('+')[1:]
        feat.append([feature_vocab['<s>']] + [feature_vocab[tag] for tag in root_and_tags] + [feature_vocab['</s>']])
        feat = torch.tensor(feat).to(args.device)    
        #  feat: (B, Tfeat)
        decoded_batch = model.reconstruct(feat)
        for sent in decoded_batch[0]:
            print("".join(sent[:-1]))

    elif args.task == 'surf2feat':
        surf = []
        surf.append([surface_vocab[char] for char in str])
        # surf: (B, Tsurf,B)
        decoded_batch = model.reconstruct(surf)
        for sent in decoded_batch[0]:
            print("".join(sent[:-1]))


task = 'feat2surf'
model = '4k_from_vae_9_9__frozendecoder_decay.pt'

#predict_tst_data(task, model)
predict(task,model,'saat+Noun+A3pl+Pnon')