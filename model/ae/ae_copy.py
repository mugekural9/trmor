# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Word copier for trained AE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from ae import AE
from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_path = 'model/ae/results/50000_instances/15epochs.pt'
model_vocab = 'model/ae/results/50000_instances/surf_vocab.json'

# logging
args.logfile = 'model/ae/results/50000_instances/15epochs_subwordcopies.txt'

# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(model_vocab) as f:
    word2id = json.load(f)
    vocab = VocabEntry(word2id)
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 512; args.nz = 32; 
args.enc_nh = 1024; args.dec_nh = 1024
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
model = AE(args, vocab, model_init, emb_init)

# load model weights
model.load_state_dict(torch.load(model_path))
model.eval()

def copy(model, word, fullword):
    x = torch.tensor([vocab.word2id['<s>']] + vocab.encode_sentence(fullword) + [vocab.word2id['</s>']]).unsqueeze(0)
    bosid = vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    # (1,1,nz)
    z = model.encoder(x)[0]
    # (1,1,dec_nh)
    c_init = model.decoder.trans_linear(z)
    h_init = torch.tanh(c_init)
    decoder_hidden = (c_init, h_init)
    copied = []; i = 0
    while True:
        # (1,1,ni)
        word_embed = model.decoder.embed(torch.tensor([input]).unsqueeze(0))
        word_embed = torch.cat((word_embed, z), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits)) 
        char = model.decoder.vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            #print(''.join(copied))
            return ''.join(copied)
            break

'''# copy word
word = "sevmiyorsun"
for i in range(len(word)):
    print('%s --> %s' % (word[:i+1],copy(model, word[:i+1])))'''

'''# copy tst data
args.tstdata = 'model/ae/data/surf.uniquesurfs.val.txt'
args.maxtstsize = 10000
tstdata = read_data(args.maxtstsize, args.tstdata, vocab, 'TST')    
with open(args.logfile, "w") as f:
    for data in tstdata:
        word =  ''.join(vocab.decode_sentence_2(data[0]))
        f.write(copy(model, word) + "\n")'''

# check subword copies
args.tstdata = 'model/ae/data/surf.uniquesurfs.val.txt'
args.maxtstsize = 1000
tstdata = read_data(args.maxtstsize, args.tstdata, vocab, 'TST')    
with open(args.logfile, "w") as f:
    for data in tstdata:
        f.write("\n-----\n")
        word =  ''.join(vocab.decode_sentence_2(data[0]))
        for i in range(len(word)):
            f.write('%s --> %s' % (word[:i+1],copy(model, word[:i+1], word[:i+1]))+"\n")
