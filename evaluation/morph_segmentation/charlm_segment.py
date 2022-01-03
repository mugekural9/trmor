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
args.type = 'prev_mid_next3'
args.eps = 0.00
# logging
args.logdir = 'evaluation/morph_segmentation/results/'+args.mname+'/'
args.logfile_meta = args.logdir + 'segments_meta_'+args.type+'_'+str(args.eps)+'.txt'
args.logfile = args.logdir + 'segments_'+args.type+'_'+str(args.eps)+'.txt'

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
    decoder_hidden = init_hidden(model,1)
    eos_probs = []
    # (1, t)
    src = model.charlm_input(x) 
    word_embed = model.embed(src)
    output, decoder_hidden = model.lstm(word_embed, decoder_hidden)
    output_logits = model.pred_linear(output).squeeze(1)
    probs = sft(output_logits)
    probs_indices = torch.argsort(probs, descending=True)
    #t
    eos_ranks = (probs_indices == eos_id).nonzero(as_tuple=True)[2].tolist()
    #t
    eos_probs = probs[:,:,2].squeeze(0).tolist()
    return  eos_ranks, eos_probs

'''# segment word into morphemes
word = 'okuyorum'
data = [1]+ [vocab[char] for char in word] 
data = torch.tensor(data).to('cuda').unsqueeze(0)
eos_ranks, eos_probs = segment(model, data)
for i in range(len(word)):
    print(word[:i+1] + ' eos_rank:' + str(eos_ranks[i]) + ' eos_prob:'+ str(eos_probs[i]))'''

def segment_with_eos_prob_average():
    fmeta = open(args.logfile_meta, "w")
    fseg = open(args.logfile, "w")
    for batch in batches:
        fmeta.write('\n----\n')
        word = ''.join(vocab.decode_sentence(batch[0][1:-1]))
        eos_ranks, eos_probs = segment(model,batch)
        avg_prob = sum(eos_probs) / len(eos_probs)
        fmeta.write('avg_prob: %.4f \n' % avg_prob)
        eos_probs_dict = dict()
        for i in range(len(word)):
            subword = word[:i+1]
            eos_probs_dict[subword] = eos_probs[i+1]

        morphemes = []
        prev_word = ''
        for k,v in eos_probs_dict.items():
            if v > avg_prob or (k==word):
                morph = k[-(len(k)-len(prev_word)):]
                morphemes.append(morph)
                prev_word = k
                fmeta.write('%s %.4f morph: %s \n' % (k,v, morph))     
        fseg.write(str(' '.join(morphemes)+'\n'))     

        for i in range(len(word)):
            #print(word[:i+1] + ' eos_rank:' + str(eos_ranks[i]) + ' eos_prob:'+ str(eos_probs[i]))
            fmeta.write('%s, eos_rank: %d, eos_prob: %.4f \n' % (word[:i+1], eos_ranks[i+1], eos_probs[i+1]))
    fmeta.close()
    fseg.close()

def segment_with_eos_prob_increase(eps):
    fmeta = open(args.logfile_meta, "w")
    fseg = open(args.logfile, "w")
    for batch in batches:
        fmeta.write('\n----\n')
        word = ''.join(vocab.decode_sentence(batch[0][1:-1]))
        eos_ranks, eos_probs = segment(model,batch)
        
        increases = []
        subword_increases = dict()
        prev_prob = 0 
        for i in range(len(word)):
            subword = word[:i+1]
            current_prob = eos_probs[i+1]
            increase = current_prob - prev_prob
            increases.append(increase)
            subword_increases[subword] = increase
            prev_prob = current_prob
            fmeta.write('%s, prob: %.4f, eos_rank: %d, increase: %.4f \n' % (word[:i+1],  current_prob, eos_ranks[i+1], increase))
        
        
        increases = increases[:-1] # exclude last
        avg_increase = sum(increases)/len(increases)
        fmeta.write('avg_increase: %.4f \n' % avg_increase)
        morphemes = []
        prev_word = ''
        for k,v in subword_increases.items():
            if v > avg_increase + eps or (k==word):
                morph = k[-(len(k)-len(prev_word)):]
                morphemes.append(morph)
                prev_word = k
                fmeta.write('%s %.4f morph: %s \n' % (k,v, morph))     
        fseg.write(str(' '.join(morphemes)+'\n'))     

        #for i in range(len(word)):
        #    #print(word[:i+1] + ' eos_rank:' + str(eos_ranks[i]) + ' eos_prob:'+ str(eos_probs[i]))
        #    fmeta.write('%s, eos_rank: %d, eos_prob: %.4f \n' % (word[:i+1], eos_ranks[i], eos_probs[i]))
    fmeta.close()
    fseg.close()

def segment_with_eos_prev_mid_next(eps):
    fmeta = open(args.logfile_meta, "w")
    fseg = open(args.logfile, "w")
    for batch in batches:
        fmeta.write('\n----\n')
        word = ''.join(vocab.decode_sentence(batch[0][1:-1]))
        eos_ranks, eos_probs = segment(model,batch)
    
        probs = []
        subword_probs = dict()
        subword_eos_ranks = dict()
        prev_prob = 0 

        for i in range(len(word)):
            subword = word[:i+1]
            current_prob = eos_probs[i+1]
            probs.append(current_prob)
            subword_probs[word[:i+1]] = current_prob
            subword_eos_ranks[word[:i+1]] = eos_ranks[i+1]
            fmeta.write('%s, eos_prob: %.4f, eos_rank: %.d   \n' % (word[:i+1],  eos_probs[i+1], eos_ranks[i+1]))

        avg_prob = sum(probs)/len(probs)
        fmeta.write('avg_prob: %.4f \n' % avg_prob)
       
        # segment to morphemes
        morphemes = []
        prev_word = ''
        subword_probs_list = [(k,v) for k,v in subword_probs.items()]
        for i in range(1,len(subword_probs_list)-1):
            prev = subword_probs_list[i-1][1]
            cur = subword_probs_list[i]
            nex = subword_probs_list[i+1][1]
            if (cur[1] > prev and cur[1] > nex): #or (cur[1] > prev  and (cur[1]-prev > nex - cur[1] + eps) and (subword_eos_ranks[cur[0]] < 10)):
                morph = cur[0][-(len(cur[0])-len(prev_word)):]
                morphemes.append(morph)
                prev_word = cur[0]
                fmeta.write('%s %.4f morph: %s \n' % (cur[0],cur[1], morph))     
        
        # add full word:
        morph = subword_probs_list[-1][0][-(len(subword_probs_list[-1][0])-len(prev_word)):]
        morphemes.append(morph)
        fmeta.write('%s %.4f morph: %s \n' % (subword_probs_list[-1][0],subword_probs_list[-1][1], morph)) 
        # write morphemes to file
        fseg.write(str(' '.join(morphemes)+'\n')) 

    fmeta.close()
    fseg.close()

segment_with_eos_prev_mid_next(args.eps)
#segment_with_eos_prob_average()