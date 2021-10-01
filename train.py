
import sys, argparse, random, torch, json, matplotlib
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
    surface_vocab, feature_vocab, pos_vocab = vocab
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
    elif args.bmodel== 'vae':
        args.basemodel = 'models/vae/vae_trmor_agg1_kls0.10_warm10_0_9.pt9'; args.model_id = '9_9'
        args.model.load_state_dict(torch.load(args.basemodel))
    elif args.bmodel== 'scratch':
        args.model_id = 'scratch'
    
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
    elif args.task == 'surf2pos':
        args.freeze_encoder = True; 
        args.freeze_decoder = False; 
        args.reset_encoder = False
        args.reset_decoder = False
        args.model.encoder.linear = None
        args.model.decoder = None
        args.model.probe_linear = nn.Linear(args.enc_nh, len(pos_vocab), bias=False)
        #args.model.pred_linear = nn.Linear(args.probedim, len(pos_vocab), bias=True)
        args.model.closs = nn.CrossEntropyLoss(reduce=False)
        args.model.vocab = vocab
    if args.bmodel == 'scratch': # fix for scratch bools 
        args.reset_encoder = True; args.reset_decoder = True
        args.freeze_encoder = False; args.freeze_decoder = False

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
    dsize = len(batches)
    indices = list(range(dsize))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf, feat, pos, root = batches[idx]
        if args.task == 'feat2surf':
            loss, _acc = args.model.s2s_loss(feat, surf)
        elif args.task == 'surf2feat':
            loss, _acc = args.model.s2s_loss(surf, feat)
        elif args.task == 'surf2pos':
            loss, _acc = args.model.linear_probe_loss(surf, pos)
        batch_loss = loss.sum()

        correct_tokens, num_tokens, wrong_tokens, wrong_predictions = _acc
        epoch_num_tokens += num_tokens
        epoch_loss += batch_loss.item()
        epoch_acc  += correct_tokens
        epoch_error += wrong_tokens

    nll = epoch_loss / dsize
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    acc = epoch_acc / epoch_num_tokens
    args.logger.write('%s --- avg_loss: %.4f, ppl: %.4f, acc: %.4f  \n' % (mode, nll,  ppl, acc))
    args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))

    return nll, ppl, acc

def train(data, args):
    trn, val, tst = data
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    #opt = optim.SGD(filter(lambda p: p.requires_grad, args.model.parameters()), lr=0.1)#, momentum=0.9)
    scheduler = ReduceLROnPlateau(opt, 'min', verbose=1, factor=0.5)
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))

    trnsize = len(trn)
    indices = list(range(trnsize))
    random.seed(0)

    best_loss = 1e4; best_ppl = 0
    trn_loss_values = []; trn_acc_values = []
    val_loss_values = []; val_acc_values = []
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = []
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad() 
            # (batchsize, t)
            surf, feat, pos, root = trn[idx] 
            if args.task == 'feat2surf':
                loss, _acc = args.model.s2s_loss(feat, surf)
            elif args.task == 'surf2feat':
                loss, _acc = args.model.s2s_loss(surf, feat)
            elif args.task == 'surf2pos':
                loss, _acc = args.model.linear_probe_loss(surf, pos)

            batch_loss = loss.sum() #mean(dim=-1)
            batch_loss.backward()
            if args.is_clip_grad:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, args.model.parameters()), args.clip_grad)
            opt.step()
            
            correct_tokens, num_tokens, wrong_tokens, wrong_predictions = _acc
            epoch_num_tokens += num_tokens
            epoch_loss       += batch_loss.item()
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions += wrong_predictions

        nll = epoch_loss / trnsize
        ppl = np.exp(epoch_loss / epoch_num_tokens)
        acc = epoch_acc / epoch_num_tokens
        error = epoch_error / epoch_num_tokens
        scheduler.step(nll)

        trn_loss_values.append(nll)
        trn_acc_values.append(acc)
        args.logger.write('\nepoch: %.1d avg_loss: %.4f, ppl: %.4f, acc: %.4f \n' % (epc, nll,  ppl, acc))
        args.logger.write('epoch correct: %.1d epoch wrong: %.1d epoch_num_tokens: %.1d \n' % (epoch_acc, epoch_error, epoch_num_tokens))

        '''
        f = open(str(epc)+"_epc.txt", "w")
        for i in epoch_wrong_predictions:
            f.write(i+'\n')
        f.close()
        '''
        
        
        # VAL
        args.model.eval()
        with torch.no_grad():
            nll, ppl, acc = test(val, "VAL", args)
            loss = nll
        val_loss_values.append(nll)
        val_acc_values.append(acc)
        #scheduler.step(nll)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            best_ppl = ppl
            torch.save(args.model.state_dict(), args.save_path)
        #torch.save(args.model.state_dict(), args.save_path)
        args.model.train()
    
    plot_curves(args.task, args.bmodel, args.fig, args.axs[0], trn_loss_values, val_loss_values, args.plt_style, 'loss')
    plot_curves(args.task, args.bmodel, args.fig, args.axs[1], trn_acc_values,  val_acc_values,  args.plt_style, 'acc')
    

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.bidirectional = False;
args.batchsize = 128; args.epochs = 1000
args.is_clip_grad = False; args.clip_grad = 5.0
args.opt= 'Adam'; args.lr = 0.005
args.task = 'surf2pos'
args.trndata = 'trmor_data/trmor2018.filtered' # 'trmor_data/trmor2018.trn'
args.valdata = 'trmor_data/trmor2018.val'
args.tstdata = 'trmor_data/trmor2018.tst'
args.seq_to_no_pad = 'surface'
args.fig, args.axs = plt.subplots(2, sharex=True)
args.trnsize = 57769 

# DATA
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
surface_vocab, feature_vocab, pos_vocab = vocab
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), 0 #len(tstdata)

# MODEL
bmodels = ['ae']#, 'vae']#, 'scratch']
plt_styles = ['-']#, '--']#, '-.']
vs_str = ''
for i,(bmodel, pstyle) in enumerate(zip(bmodels, plt_styles)): 
    # Logging
    args.bmodel = bmodel 
    args.plt_style = pstyle
    args.modelname = str(args.trnsize)+'instances_from_'+args.bmodel+'_'+str(args.epochs) + 'epochs'
    args.save_path = 'models/' + args.task + '/'+ args.modelname + '.pt'
    args.log_path = 'logs/' + args.task + '/' + args.modelname + '.log'
    args.fig_path = 'figs/' + args.task +'/'+ args.modelname + '.png'
    args.logger = Logger(args.log_path)
    log_data(trndata, 'trn', surface_vocab, feature_vocab, pos_vocab, args.logger)
    log_data(vlddata, 'val', surface_vocab, feature_vocab, pos_vocab, args.logger)
    args.enc_dropout_in = 0.0
    args.dec_dropout_in = 0.0
    args.dec_dropout_out = 0.0
    args.model = build_model(args, vocab, rawdata)
    args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
    args.logger.write(args)
    args.logger.write('\n')

    train(batches, args)
    vs_str+= bmodel
    if i+1 < len(bmodels):
      vs_str+= '_vs_'

plt.savefig(args.fig_path)
