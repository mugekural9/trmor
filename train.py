import random
import torch
import torch.nn as nn
from torch import optim
from data import readdata, get_batches
from model import EncoderRNN, DecoderRNN, loss, test, pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

# Read data and get batches...
data, vocab = readdata()  # data len:848723
trndata = data[:830000]       # 830000
vlddata = data[830000:840000] # 10000
tstdata = data[840000:]       # 8723
trn_batches, _ = get_batches(trndata, vocab, batchsize=64) 
vld_batches, _ = get_batches(vlddata, vocab, batchsize=64) 
tst_batches, _ = get_batches(tstdata, vocab, batchsize=1) 

# Build model...
hiddensize = 128
vocabsize = len(vocab.word2idx) # 287
encoder = EncoderRNN(vocabsize, hiddensize).to(device)
decoder = DecoderRNN(hiddensize, vocabsize).to(device)

# Train...
criterion = nn.NLLLoss(ignore_index=0)
learning_rate = 0.001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
epochs = 1
for epoch in range(epochs):
    print('\n -----epoch: ', epoch)
    epoch_loss = 0
    epoch_acc = 0
    indices = list(range(len(trn_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_tensor, target_tensor = trn_batches[idx]
        batch_loss, batch_acc = loss(input_tensor.to(device), target_tensor.to(device), encoder, decoder, criterion)
        batch_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        epoch_loss += batch_loss
        epoch_acc  += batch_acc
    print('TRN epoch_loss:', epoch_loss/len(trn_batches))
    print('TRN epoch_acc:', epoch_acc/len(trn_batches))
    
    # Valid...
    epoch_loss = 0
    epoch_acc = 0
    indices = list(range(len(vld_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        input_tensor, target_tensor = vld_batches[idx]
        batch_loss, batch_acc = loss(input_tensor.to(device), target_tensor.to(device), encoder, decoder, criterion, vocab=vocab, is_report=False)
        epoch_loss += batch_loss
        epoch_acc  += batch_acc
    print('VAL epoch_loss:', epoch_loss/len(vld_batches))
    print('VAL epoch_acc:', epoch_acc/len(vld_batches))


# Test...
epoch_loss = 0
epoch_acc = 0
indices = list(range(len(tst_batches)))
random.shuffle(indices)
for i, idx in enumerate(indices):
    input_tensor, target_tensor = tst_batches[idx]
    batch_loss, batch_acc = test(input_tensor.to(device), target_tensor.to(device), encoder, decoder, criterion, vocab)
    epoch_loss += batch_loss
    epoch_acc  += batch_acc
print('TST epoch_loss:', epoch_loss/len(tst_batches))
print('TST epoch_acc:', epoch_acc/len(tst_batches))
   

# Pred...
input = torch.tensor([ 1, 43,  5,  8, 27,  5, 12, 35, 15, 22,  2]).unsqueeze(1).to(device)
input_tokens=[]
for t in input[:,0]:
    tok = vocab.idx2word[t]
    input_tokens.append(tok)
output_tokens = pred(input, encoder, decoder, vocab)
print('INPUT:', input_tokens)
print('OUTPUT:', output_tokens)
