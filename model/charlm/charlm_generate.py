# -----------------------------------------------------------
# Date:        2021/12/17 
# Author:      Muge Kural
# Description: Random word generator for trained charLM model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from charlm import CharLM
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_path  = 'model/charlm/results/582000_instances/35epochs.pt'
model_vocab = 'model/charlm/results/582000_instances/surf_vocab.json'

# initialize model
# load vocab (to initialize the model with correct vocabsize)
with open(model_vocab) as f:
    word2id = json.load(f)
    vocab = VocabEntry(word2id)
model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
args.ni = 64; args.nh = 350; args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
model = CharLM(args, vocab, model_init, emb_init) 

# load model weights
model.load_state_dict(torch.load(model_path))
model.eval()

def init_hidden(model, bsz):
    weight = next(model.parameters())
    return (weight.new_zeros(1, bsz, model.nh),
            weight.new_zeros(1, bsz, model.nh))

def generate(model):
    bosid = vocab.word2id['<s>'] 
    input = torch.tensor([bosid]).unsqueeze(0)
    word = []
    sft = nn.Softmax(dim=1)
    i = 0; max_length = 20
    decoder_hidden = init_hidden(model,1)
    while i < max_length:
        i +=1
        word_embed = model.embed(input)
        output, decoder_hidden = model.lstm(word_embed, decoder_hidden)
        output_logits = model.pred_linear(output).squeeze(1)
        input = torch.multinomial(sft(output_logits), num_samples=1) # sample
        char = model.vocab.id2word(input.item())
        word.append(char)
        if char == '</s>':
            print(''.join(word))
            break

# generate random words
for i in range(100):
    generate(model)


