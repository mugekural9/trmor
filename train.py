import random
import time
import torch
import torch.nn as nn
from torch import optim
from data import readdata, get_batches
from model import TextVAE
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for batch_data in test_data_batch:
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples


def epoch(batches, vocab, model, opt='None', mode='trn', kl_anneal=1, teac_forc=False):
    if mode != 'trn': 
        model.eval()
    else:
        model.train()            
        
    epoch_loss = 0; epoch_kl_loss = 0; epoch_rec_loss = 0; epoch_align_loss = 0
    epoch_acc = 0
    indices = list(range(len(batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        if mode == 'trn':
            opt.zero_grad()
        input_tensor, target_tensor = batches[idx]
        #if type=='ae':
        #    input_tensor = target_tensor # ae
        batch_loss,  rec_loss, align_loss, kl_loss, batch_acc = model.loss(input_tensor.to(device), target_tensor.to(device), vocab, mode, teac_forc, beta=beta) 
        if mode == 'trn':
            batch_loss.backward()
            opt.step()
        epoch_loss += batch_loss.item()
        epoch_acc  += batch_acc
        epoch_rec_loss  += rec_loss.item()
        epoch_align_loss += align_loss.item()
        epoch_kl_loss += kl_loss.item()
    return epoch_loss/len(batches), epoch_rec_loss/len(batches), epoch_align_loss/len(batches), epoch_kl_loss/len(batches), epoch_acc /len(batches)


def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for data in test_data_batch:
        batch_data = data[1]
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples


def train(num_epochs, vocab, model, opt):
    best_model = None
    best_trn = 0
    best_val = 0
    best_tst = 0
    for epc in range(num_epochs):
        print('\n -----epoch: ', epc)
            
        epoch_loss, epoch_rec_loss, epoch_align_loss, epoch_kl_loss, epoch_acc = epoch(trn_batches, vocab, model, opt=opt, mode='trn', teac_forc=False)
        print('TRN epoch_loss:', epoch_loss, 'TRN epoch_rec_loss:', epoch_rec_loss, 'TRN epoch_align_loss:', epoch_align_loss,'TRN epoch_kl_loss:', epoch_kl_loss, 'TRN epoch_acc:', epoch_acc)
        log = 'epoch: '+ str(epc) + ' epoch_loss:' + str(epoch_loss) +  ' epoch_rec_loss:' + str(epoch_rec_loss) + ' epoch_acc:' + str(epoch_acc)
        writer.add_text('1_trn', log)
        writer.add_scalar('Loss/1_trn', epoch_loss, epc)
        writer.add_scalar('Accuracy/1_trn', epoch_acc, epc)
        if epoch_acc > best_trn:
            best_trn = epoch_acc
       
        # epoch_loss, epoch_rec_loss, epoch_align_loss, epoch_kl_loss,  epoch_acc = epoch(vld_batches, vocab, model, mode='val', teac_forc=False)
        # print('VAL epoch_loss:', epoch_loss, 'VAL epoch_rec_loss:', epoch_rec_loss, 'VAL align_loss:', epoch_align_loss, 'VAL epoch_kl_loss:', epoch_kl_loss, 'VAL epoch_acc:', epoch_acc)
        # log = 'epoch: '+ str(epc) + ' epoch_loss:' + str(epoch_loss) +  ' epoch_rec_loss:' + str(epoch_rec_loss) + ' epoch_acc:' + str(epoch_acc)
        # writer.add_text('2_val', log)
        # writer.add_scalar('Loss/2_val', epoch_loss, epc)
        # writer.add_scalar('Accuracy/2_val', epoch_acc, epc)

        # if epoch_acc > best_val:
        #     best_val = epoch_acc
            
        epoch_loss, epoch_rec_loss, epoch_align_loss, epoch_kl_loss, epoch_acc = epoch(tst_batches, vocab, model, mode='tst', teac_forc=False)
        print('TST epoch_loss:', epoch_loss, 'TST epoch_rec_loss:', epoch_rec_loss, 'TST epoch_align_loss:', epoch_align_loss, 'TST epoch_kl_loss:', epoch_kl_loss, 'TST epoch_acc:', epoch_acc)
        mi = calc_mi(model, tst_batches)
        print('mi:',mi)
        if epoch_acc > best_tst:
            best_tst = epoch_acc
            best_model = model
        log = 'epoch: '+ str(epc) + ' epoch_loss:' + str(epoch_loss) +  ' epoch_rec_loss:' + str(epoch_rec_loss) + ' epoch_acc:' + str(epoch_acc)
        writer.add_text('3_tst', log)
        writer.add_scalar('Loss/3_tst', epoch_loss, epc)
        writer.add_scalar('Accuracy/3_tst', epoch_acc, epc)

    _best_val = str(best_val)
    _best_tst = str(best_tst)
    print('bestval:'+ _best_val)
    print('besttst:'+ _best_tst)
    writer.add_text('bestval', _best_val)
    writer.add_text('besttst', _best_tst)
    return best_model, best_val, best_tst, best_trn



def log_data_info(data, dset):
    f = open(dset+".txt", "w")
    total_inp_len = 0
    total_out_len = 0
    for d in data:
        inpdata = d[1]
        outdata = d[0]
        total_inp_len += len(inpdata)
        total_out_len += len(outdata)
        f.write(outdata+'\n')

    f.close()
    avg_inp_len = total_inp_len / len(data)
    avg_out_len = total_out_len / len(data)

    writer.add_text(dset+' DATA', 'datasize:'+ str(len(data)) +' avg_inp_len, avg_out_len:'+ str(avg_inp_len) + ' ' + str(avg_out_len))
    print(dset+' avg_inp_len:', str(avg_inp_len)+' , avg_out_len:' + str(avg_out_len))
    
    
# # Read data and get batches...
data, vocab = readdata()     # data len:69981

trndata = data[:60000]       # 60000
vlddata = data[60000:65000]  # 5000
tstdata = data[65000:]  # 4981

log_data_info(trndata, 'trn')
log_data_info(vlddata, 'val')
log_data_info(tstdata, 'tst')
batchsize = 256
trn_batches, _ = get_batches(trndata, vocab, batchsize) 
vld_batches, _ = get_batches(vlddata, vocab, batchsize) 
tst_batches, _ = get_batches(tstdata, vocab, batchsize) 
# Build and train model...
modeltype = 'vae'
hiddensize = 512
latent_dim = 32
vocabsize = len(vocab.word2idx)
model = TextVAE(hiddensize, latent_dim, vocabsize, type=modeltype).to(device)



bests_trn = []
bests_val = []
bests_tst = []
all_best_tst = 0
repeat=1
for i in range(repeat):
    print('################### i: ',str(i))
    modeltype = 'vae'
    basetype  = 'vae'
    beta = 1
   
    if modeltype == 'ft':
        modeltype += '-' + basetype
        
    model = TextVAE(hiddensize, latent_dim, vocabsize, type=modeltype).to(device)
    print('model:', model)
    print('trndatasize:', len(trndata), 'vlddatasize:', len(vlddata), 'tstdatasize:', len(tstdata), ' batchsize:', batchsize)
    timestr = time.strftime("%Y%m%d-%H%M%S")   
    # Train...
    opt = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200; Nkl = 10
    
    if 'ft' in model.type:
        basedmodelname = 'ft-vae_1_5.0k_5.pt'
        #str(beta) + '_' + basetype+ str(800.0)+ 'k_' + str(100) + '.pt'
        model.load_state_dict(torch.load(basedmodelname))
        model.E_s2s = nn.LSTM(hiddensize, 16, 1, dropout=0.0, bidirectional=True).to(device)
        modelname = model.type +'_'+str(beta)+'_' +str(len(trndata)/1000)+'k_'+str(num_epochs)+'.pt'
        for param in model.parameters():
            param.requires_grad = False
        for param in model.embed_s2s.parameters():
            param.requires_grad = True
        for param in model.E_s2s.parameters():
            param.requires_grad = True
    else:
        modelname = model.type + str(len(trndata)/1000)+'k_'+str(num_epochs)+'.pt'
        if model.type == 'vae':
            modelname = str(beta) +'_' +modelname

    print('Model type: ', model.type)
    model, best_val, best_tst, best_trn = train(num_epochs, vocab, model, opt)    
    bests_trn.append(best_trn)
    bests_val.append(best_val)
    bests_tst.append(best_tst)
    if best_tst > all_best_tst:
        torch.save(model.state_dict(), modelname)
        all_best_tst = best_tst
    writer.add_text('model', modelname+'----- '+str(model))
    writer.add_text('configs:', 'batchsize:' + str(batchsize) + ' num_epochs: ' + str(num_epochs) + ' opt: ' + str(opt))

if repeat > 1:
    print('---')
    print('best trns:' , bests_trn)
    print('best vals:' , bests_val)
    print('best tsts:' , bests_tst)
    print('bests trn in:'+str(repeat), max(bests_trn))
    print('bests val in:'+str(repeat), max(bests_val))
    print('bests tst in:'+str(repeat), max(bests_tst))
    writer.add_text(str(repeat)+'models_trn: ', str(bests_trn))
    writer.add_text(str(repeat)+'models_val: ', str(bests_val))
    writer.add_text(str(repeat)+'models_tst: ', str(bests_tst))

# Eval...
#model.load_state_dict(torch.load('models/model_ae_without_tf.pt'))
#model.load_state_dict(torch.load('1_vae60.0k_200.pt'))
#epoch_loss, epoch_kl_loss, epoch_rec_loss, epoch_acc = epoch(tst_batches, vocab, model, mode='tst', teac_forc=False)
#print('TST epoch_loss:', epoch_loss, 'TST epoch_kl_loss:', epoch_kl_loss, 'TST epoch_rec_loss:', epoch_rec_loss, 'TST epoch_acc:', epoch_acc)
# writer.close()














######################


# # # Sampling and interpolation...
# #for i in range(100):
# generation1 = []
# sample_gen, z1 = model.sample_from_prior()
# for tok in sample_gen:
#     if vocab.idx2word[tok]!= '<eos>': 
#         generation1.append(vocab.idx2word[tok])

# generation2 = []
# sample_gen, z2 = model.sample_from_prior()
# for tok in sample_gen:
#     if vocab.idx2word[tok]!= '<eos>':
#         generation2.append(vocab.idx2word[tok])
# print('GENERATED z1:', ''.join(generation1), ' GENERATED z2:', ''.join(generation2))

# import numpy as np
# n=10
# z3 = []
# for i in range(n):
#     zi = torch.lerp(z1, z2, 1.0*i/(n-1))
#     z3.append(np.expand_dims(zi.cpu(), axis=0))
# z3= torch.tensor(np.concatenate(z3, axis=0)).to(device)
# for i in range(10):
#     generation = []
#     sample_gen, _ = model.sample_from_prior(z3[i])
#     for tok in sample_gen:
#         if vocab.idx2word[tok]!= '<eos>':
#             generation.append(vocab.idx2word[tok])
#     print(''.join(generation))


# # Pred...
# input = torch.tensor([ 1, 43,  5,  8, 27,  5, 12, 35, 15, 22,  2]).unsqueeze(1).to(device)
# input_tokens=[]
# for t in input[:,0]:
#    tok = vocab.idx2word[t]
#    input_tokens.append(tok)
#    output_tokens = pred(input, encoder, decoder, vocab)
# print('INPUT:', input_tokens)
# print('OUTPUT:', output_tokens)


# Pred...
# input = torch.tensor([  1, 13,  5, 38, 39,  7, 24,  6,  2 ]).unsqueeze(1).to(device)
# input_tokens  = []
# output_tokens = []
# for t in input:
#     tok = vocab.idx2word[t]
#     input_tokens.append(tok)
# _, _, output = model.pred(input, vocab)
# for t in output:
#     tok = vocab.idx2word[t]
#     if tok == '<eos>':
#         break
#     output_tokens.append(tok)
# print('INPUT:', input_tokens)
# print('OUTPUT:', output_tokens)

