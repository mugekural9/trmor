# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Word morpheme segmentation for trained charLM model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from model.charlm.charlm import CharLM
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
from data.data import build_data

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
model_path  = 'model/charlm/results/582000_instances/35epochs.pt'
model_vocab = 'model/charlm/results/582000_instances/surf_vocab.json'
args.mname = 'charlm'

# logging
args.logdir = 'evaluation/morph_segmentation/results/'+args.mname+'/'
args.logfile = args.logdir + 'segments.txt'
try:
    os.makedirs(args.logdir)
    print("Directory " , args.logdir ,  " Created ") 
except FileExistsError:
    print("Directory " , args.logdir ,  " already exists")


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
model.to(args.device)
model.eval()

# data
args.tstdata = 'evaluation/morph_segmentation/data/goldstdsample.tur'
args.maxtstsize = 1000
data, batches = build_data(args, vocab)

def init_hidden(model, bsz):
    weight = next(model.parameters())
    return (weight.new_zeros(1, bsz, model.nh),
            weight.new_zeros(1, bsz, model.nh))


def segment(model, x):
    eos_id = vocab.word2id['</s>']
    sft = nn.Softmax(dim=2)
    i = 0; max_length = 20
    decoder_hidden = init_hidden(model,1)
    eos_probs = []
    word_embed = model.embed(x)
    output, decoder_hidden = model.lstm(word_embed, decoder_hidden)
    output_logits = model.pred_linear(output).squeeze(1)
    probs = sft(output_logits)
    probs_indices = torch.argsort(probs, descending=True)
    eos_ranks = (probs_indices == eos_id).nonzero(as_tuple=True)[2].tolist()
    eos_probs = probs[:,:,2].squeeze(0).tolist()
    return  eos_ranks[1:], eos_probs[1:]

'''# segment word into morphemes
word = 'okuyorum'
data = [1]+ [vocab[char] for char in word] 
data = torch.tensor(data).to('cuda').unsqueeze(0)
eos_ranks, eos_probs = segment(model, data)
for i in range(len(word)):
    print(word[:i+1] + ' eos_rank:' + str(eos_ranks[i]) + ' eos_prob:'+ str(eos_probs[i]))'''

f = open(args.logfile, "w")
for batch in batches:
    f.write('\n----\n')
    word = ''.join(vocab.decode_sentence(batch[0][1:-1]))
    eos_ranks, eos_probs = segment(model,batch)
    for i in range(len(word)):
        #print(word[:i+1] + ' eos_rank:' + str(eos_ranks[i]) + ' eos_prob:'+ str(eos_probs[i]))
        f.write('%s, eos_rank: %.4f, eos_prob: %.4f \n' % (word[:i+1], eos_ranks[i], eos_probs[i]))
f.close()
