
import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from modules import VAE, LSTMEncoder, LSTMDecoder
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from data import build_data, log_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def build_model(args, vocab, rawdata):
    surface_vocab, feature_vocab, pos_vocab, polar_vocab, tense_vocab = vocab
    # MODEL
    # must be that way to load pretrained ae and vae
    args.ni = 512
    args.enc_nh = 1024
    args.dec_nh = 1024
    args.nz = 32
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
    decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  #feature_vocab for scratch preloading
    args.model = VAE(encoder, decoder, args)
    args.model.encoder.is_reparam = False

    # Switch base model
    if args.bmodel== 'ae':
        args.basemodel = 'models/ae/ae_trmor_agg0_kls0.10_warm10_0_9.pt9';  args.model_id = '9_9'
        args.model.load_state_dict(torch.load(args.basemodel))
    elif args.bmodel == 'vae' or args.bmodel == 'vae+probe':
        args.basemodel = 'models/vae/vae_trmor_agg1_kls0.10_warm10_0_9.pt9'; args.model_id = '9_9'
        args.model.load_state_dict(torch.load(args.basemodel))
    elif args.bmodel== 'scratch':
        args.model_id = 'scratch'
    elif args.bmodel== 'random':
        args.model_id = 'random'

    # configure task related components
    if args.task   == 'feat2surf':
        args.freeze_encoder = False; 
        args.freeze_decoder = True; 
        args.reset_encoder = True
        args.reset_decoder = False
        args.model.encoder = LSTMEncoder(args, len(feature_vocab.word2id), model_init, emb_init, bidirectional=args.bidirectional)
        args.model.encoder.is_reparam = False
    elif args.task == 'surf2feat':
        args.freeze_encoder = True; 
        args.freeze_decoder = False; 
        args.reset_encoder = False
        args.reset_decoder = True
        args.model.decoder = LSTMDecoder(args, feature_vocab, model_init, emb_init)
    elif args.task == 'surf2pos' or args.task == 'root2pos':
        args.freeze_encoder = True 
        args.freeze_decoder = True
        args.reset_encoder = False
        args.reset_decoder = False
        args.model.encoder.linear = None
        args.model.decoder = None
        args.model.probe_linear = nn.Linear(args.enc_nh, len(pos_vocab), bias=False)
        args.model.closs = nn.CrossEntropyLoss(reduce=False)
        args.model.vocab = vocab
    elif args.task == 'surf2polar':
        args.freeze_encoder = True 
        args.freeze_decoder = True 
        args.reset_encoder = False
        args.reset_decoder = False
        args.model.encoder.linear = None
        args.model.decoder = None
        args.model.probe_linear = nn.Linear(args.enc_nh, len(polar_vocab), bias=False)
        args.model.closs = nn.CrossEntropyLoss(reduce=False)
        args.model.vocab = vocab
    elif args.task == 'surf2tense':
        args.freeze_encoder = True
        args.freeze_decoder = True
        args.reset_encoder = False
        args.reset_decoder = False
        args.model.encoder.linear = None
        args.model.decoder = None
        args.model.probe_linear = nn.Linear(args.enc_nh, len(tense_vocab), bias=False)
        args.model.closs = nn.CrossEntropyLoss(reduce=False)
        args.model.vocab = vocab

    if args.bmodel == 'scratch': # fix for scratch bools 
        args.reset_encoder = True; args.reset_decoder = True
        args.freeze_encoder = False; args.freeze_decoder = False
    if args.bmodel == 'random': # fix for random bools 
        args.reset_encoder = True; args.reset_decoder = True
        args.freeze_encoder = True; args.freeze_decoder = True
    if args.bmodel == 'vae+probe':
        args.model.load_state_dict(torch.load('logs/surf2pos/vae/47500_instances/300epochs.pt'))
    if args.bmodel == 'ae+probe':
        args.model.load_state_dict(torch.load('logs/surf2pos/ae/47500_instances/300epochs.pt'))   
    # reset and freeze parameters if needed
    if args.reset_encoder:
        if args.model.encoder is not None:
            args.model.encoder.reset_parameters(model_init, emb_init)
    if args.reset_decoder: 
        if args.model.decoder is not None:
            args.model.decoder.reset_parameters(model_init, emb_init)
    if args.freeze_encoder:
        if args.model.encoder is not None:
            for param in args.model.encoder.parameters():
                param.requires_grad = False
    if args.freeze_decoder:
        if args.model.decoder is not None:
            for param in args.model.decoder.parameters():
                param.requires_grad = False

    args.model.to(args.device)  
    return args.model

def test(batches, mode, args):
    epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
    epoch_wrong_predictions = [];
    epoch_correct_predictions = [];
    dsize = len(batches)
    indices = list(range(dsize))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, feat, pos, root, polar, tense = batches[idx] 
        if args.task == 'feat2surf':
            loss, _acc = args.model.s2s_loss(feat, surf)
        elif args.task == 'surf2feat':
            loss, _acc = args.model.s2s_loss(surf, feat)
        elif args.task == 'surf2pos':
            loss, _acc, _ = args.model.linear_probe_loss(surf, pos, args.freqdict, args.freqstagsdict)
        elif args.task == 'root2pos':
            loss, _acc, _ = args.model.linear_probe_loss(root, pos, args.freqdict, args.freqstagsdict)
        elif args.task == 'surf2polar':
            loss, _acc, _ = args.model.linear_probe_loss(surf, polar, args.freqdict, args.freqstagsdict)
        elif args.task == 'surf2tense':
            loss, _acc, _ = args.model.linear_probe_loss(surf, tense, args.freqdict, args.freqstagsdict)
        batch_loss = loss.sum()
        correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
        epoch_num_tokens += num_tokens
        epoch_loss += batch_loss.item()
        epoch_acc  += correct_tokens
        epoch_error += wrong_tokens
        epoch_wrong_predictions += wrong_predictions
        epoch_correct_predictions += correct_predictions
    nll = epoch_loss / dsize
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f  \n' % (mode, nll,  ppl, acc))
    args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
    f1 = open(args.modelname + "/"+mode+"_wrong_predictions.txt", "w")
    f2 = open(args.modelname + "/"+mode+"_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        f1.write(i+'\n')
    for i in epoch_correct_predictions:
        f2.write(i+'\n')
    f1.close(); f2.close()
    return nll, ppl, acc

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4; best_ppl = 0
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = [];
        epoch_correct_predictions = [];
        general_last_states = []
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, feat, pos, root, polar, tense = trnbatches[idx] 
            if args.task == 'feat2surf':
                loss, _acc = args.model.s2s_loss(feat, surf)
            elif args.task == 'surf2feat':
                loss, _acc = args.model.s2s_loss(surf, feat)
            elif args.task == 'surf2pos':
                loss, _acc, general_last_states = args.model.linear_probe_loss(surf, pos, args.freqdict, args.freqstagsdict)
            elif args.task == 'surf2polar':
                loss, _acc, general_last_states = args.model.linear_probe_loss(surf, polar, args.freqdict, args.freqstagsdict)
            elif args.task == 'surf2tense':
                loss, _acc, general_last_states = args.model.linear_probe_loss(surf, tense, args.freqdict, args.freqstagsdict)
            batch_loss = loss.sum() #mean(dim=-1)
            batch_loss.backward()
            if args.is_clip_grad:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, args.model.parameters()), args.clip_grad)
            opt.step()
            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
            epoch_num_tokens += num_tokens
            epoch_loss       += batch_loss.item()
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions += wrong_predictions
            epoch_correct_predictions += correct_predictions
        nll = epoch_loss / numbatches
        ppl = np.exp(epoch_loss / epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        error = epoch_error / epoch_num_tokens
        scheduler.step(nll)
        # (nist, dim)
        # X = torch.cat(general_last_states).cpu().detach()
        # find_li_vectors(1024,X)
        # breakpoint()
        # rank_X = np.linalg.matrix_rank(X) # rank-defficient
        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, ppl: %.4f, acc: %.4f \n' % (epc, nll,  ppl, acc))
        args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
        f1 = open(args.modelname + "/trn_wrong_predictions.txt", "w")
        f2 = open(args.modelname + "/trn_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, ppl, acc = test(valbatches, "val", args)
            loss = nll
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            best_ppl = ppl
            torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    plot_curves(args.task, args.bmodel, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.bmodel, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.bidirectional = False;
args.is_clip_grad = False; args.clip_grad = 5.0
args.batchsize = 128; args.epochs = 300
args.opt= 'Adam'; args.lr = 0.01
args.task = 'surf2pos'
args.seq_to_no_pad = 'surface'
args.trndata = 'trmor_data/pos/pos.trn.txt' # 'trmor_data/trmor2018.trn'
args.valdata = 'trmor_data/pos/pos.val.txt'
args.tstdata = 'trmor_data/pos/pos.uniqueroots.txt'
args.fig, args.axs = plt.subplots(2, sharex=True)
args.maxtrnsize = 57769 

# TRAIN
bmodels = ['random']#, 'vae']#, 'scratch']
plt_styles = ['-']#, '--']#, '-.']
vs_str = ''
for i,(bmodel, pstyle) in enumerate(zip(bmodels, plt_styles)): 
    # data
    args.bmodel = bmodel 
    args.plt_style = pstyle
    rawdata, batches, vocab = build_data(args)
    trndata, vlddata, tstdata = rawdata
    args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), 0
    # logging
    args.save_path = args.modelname + '/'+ str(args.epochs)+'epochs.pt'
    args.log_path =  args.modelname + '/'+ str(args.epochs)+'epochs.log'
    args.fig_path =  args.modelname + '/'+ str(args.epochs)+'epochs.png'
    args.logger = Logger(args.log_path)
    log_data(trndata, 'trn', vocab, args.logger,args.modelname)
    log_data(vlddata, 'val', vocab, args.logger,args.modelname, 'val')
    log_data(tstdata, 'tst', vocab, args.logger,args.modelname, 'tst')
    # model
    args.enc_dropout_in = 0.0
    args.dec_dropout_in = 0.0
    args.dec_dropout_out = 0.0
    args.model = build_model(args, vocab, rawdata)
    args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
    args.logger.write(args)
    args.logger.write('\n')
    train(batches, args)
    '''
    trnbatches, valbatches, tstbatches = batches
    args.model.eval()
    with torch.no_grad():
        test(tstbatches, "tst", args)
    '''
    vs_str+= bmodel
    if i+1 < len(bmodels):
      vs_str+= '_vs_'
plt.savefig(args.fig_path)
