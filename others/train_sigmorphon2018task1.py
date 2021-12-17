
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
    surface_vocab, feature_vocab = vocab
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
    if args.bmodel== 'ae_original':
        args.basemodel = 'models/ae/trmor_agg0_kls0.00_warm10_0_911.pt'
        args.model.load_state_dict(torch.load(args.basemodel))
    elif args.bmodel == 'vae_original_2':
        args.basemodel = 'models/vae/trmor_agg0_kls0.10_warm10_0_911.pt'
        args.model.load_state_dict(torch.load(args.basemodel))

 
    # configure task related components
    args.freeze_encoder = False; 
    args.freeze_decoder = True; 
    args.reset_encoder = True
    args.reset_decoder = False
    args.model.encoder = LSTMEncoder(args, len(feature_vocab.word2id), model_init, emb_init, bidirectional=args.bidirectional)
    args.model.encoder.is_reparam = False

    if args.bmodel == 'scratch': # fix for scratch bools 
        args.reset_encoder = True; args.reset_decoder = True
        args.freeze_encoder = False; args.freeze_decoder = False
    if args.bmodel == 'random': # fix for random bools 
        args.reset_encoder = True; args.reset_decoder = True
        args.freeze_encoder = True; args.freeze_decoder = True
    
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
        surf, feat = batches[idx] 
        # loss: (batchsize)
        loss, _acc = args.model.s2s_loss(feat, surf)
        batch_loss = loss.sum()
        correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
        epoch_num_tokens += num_tokens
        epoch_loss       += batch_loss.item()
        epoch_acc        += correct_tokens
        epoch_error      += wrong_tokens
        epoch_wrong_predictions += wrong_predictions
        epoch_correct_predictions += correct_predictions

    nll = epoch_loss / args.valsize
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f  \n' % (mode, nll,  ppl, acc))
    args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
    f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_"+ mode + "_wrong_predictions.txt", "w")
    f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_"+ mode + "_correct_predictions.txt", "w")
    for i in epoch_wrong_predictions:
        f1.write(i+'\n')
    for i in epoch_correct_predictions:
        f2.write(i+'\n')
    f1.close(); f2.close()
    return nll, ppl, acc

def train(data, args):
    trnbatches, valbatches, _ = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5, patience=7)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4; best_ppl = 1e4; best_acc = 0
    not_improved = 0
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = [];
        epoch_correct_predictions = [];
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, feat = trnbatches[idx] 
            # loss: (batchsize)
            loss, _acc = args.model.s2s_loss(feat, surf)
            batch_loss = loss.sum()
            batch_loss.backward()
            opt.step()
            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
            epoch_num_tokens += num_tokens
            epoch_loss       += batch_loss.item()
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions += wrong_predictions
            epoch_correct_predictions += correct_predictions
        nll = epoch_loss / args.trnsize
        ppl = np.exp(epoch_loss / epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens

        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, ppl: %.4f, acc: %.4f \n' % (epc, nll,  ppl, acc))
        args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))
        f1 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
        f2 = open(args.modelname + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
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
        scheduler.step(nll)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            torch.save(args.model.state_dict(), args.save_path)
            not_improved = 0
        if ppl < best_ppl:
            best_ppl = ppl
        if acc > best_acc:
            best_acc = acc
        else:
            not_improved += 1
        
        args.model.train()
        if not_improved > 42:
            args.logger.write('Best LOSS was: %.4f \n' % best_loss)
            args.logger.write('Best PPl was:  %.4f \n' % best_ppl)
            args.logger.write('Best ACC was:  %.4f \n' % best_acc)
            break
       
    plot_curves(args.task, args.bmodel, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.bmodel, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.bidirectional = False;
args.is_clip_grad = False; args.clip_grad = 5.0
args.batchsize = 128; args.epochs = 100
args.opt= 'Adam'; args.lr = 0.001
args.task = 'sigmorphon2018task1/low-data'
args.seq_to_no_pad = 'surface'
args.trndata = 'trmor_data/sigmorphon/2018task-1/turkish-train-low' 
args.valdata = 'trmor_data/sigmorphon/2018task-1/turkish-dev' 
args.tstdata = 'trmor_data/sigmorphon/2018task-1/turkish-test' 
args.surface_vocab_file = 'trmor_data/sigmorphon/2018task-1/turkish-train-high' 

args.fig, args.axs = plt.subplots(2, sharex=True)
args.maxtrnsize = 10000 
args.maxvalsize = 8329 
args.maxtstsize = 8517

# TRAIN
bmodels = ['vae_original_2']#, 'vae']#, 'scratch']
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
    
    '''trnbatches, valbatches, tstbatches = batches
    args.model.eval()
    with torch.no_grad():
        test(tstbatches, "tst", args)'''
    
    vs_str+= bmodel
    if i+1 < len(bmodels):
      vs_str+= '_vs_'
plt.savefig(args.fig_path)
