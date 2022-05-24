# -----------------------------------------------------------
# Date:        2022/02/01 
# Author:      Muge Kural
# Description: Morpheme segmentation heuristics for trained charlm model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from model.vqvae.vqvae import VQVAE
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
import numpy as np
from data.data import build_data
from collections import OrderedDict

# heur3: detects morpheme boundary if: 
#        (1) the model cannot copy the word correctly (indicating that the sub word is not valid)
def heur_check_copying(args, word):
    def copy(args, word):
        x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
        bosid = args.vocab.word2id['<s>']
        input = torch.tensor(bosid)
        sft = nn.Softmax(dim=1)
        quantized_inputs, vq_loss, quantized_inds, encoder_fhs = args.model.vq_loss(x, 0)
        root_z, suffix_z = quantized_inputs
        c_init = root_z
        h_init = torch.tanh(c_init)
        decoder_hidden = (c_init, h_init)
        copied = []; i = 0
        while True:
            # (1,1,ni)
            word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0))
            word_embed = torch.cat((word_embed, suffix_z), -1)
            # output: (1,1,dec_nh)
            output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
            # (1, vocab_size)
            output_logits = args.model.decoder.pred_linear(output).squeeze(1)
            input = torch.argmax(sft(output_logits)) 
            char = args.vocab.id2word(input.item())
            copied.append(char)
            if char == '</s>':
                #print(''.join(copied))
                return ''.join(copied), quantized_inds
            
    morphemes = []
    prev_word = ''
    root_id = copy(args, word)[1][0][0][0].item()
    for i in range(1,len(word)+1):
        copied_word, subword_root_id = copy(args, word[:i])
        subword_root_id = subword_root_id[0][0][0].item()
        if subword_root_id == root_id and word[:i]+'</s>' == copied_word and len(word[:i]) >2:
            morphemes.append(word[:i][len(prev_word):])
            prev_word = word[:i]

    # add full word if missing
    last_morph = word[len(prev_word):]
    if len(last_morph) != 0:
        morphemes.append(last_morph)
    #print(morphemes)
    return morphemes


# heur2: detects morpheme boundary if: 
#        (1) the current likelihood(ll) exceeds prev and next ll OR current ll increase excesses prev inc
def heur_prev_mid_next_and_prevnext_exceed(logps, eps):
    morphemes = []
    prev_word = ''
    logps = [(k,v) for k,v in logps.items()]
    for i in range(1,len(logps)-1):
        prev = logps[i-1][1]
        cur = logps[i]
        nex = logps[i+1][1]
        if (i>0 and (cur[1] > (prev + nex)/2)  and len(cur[0])>2) or (cur[1] > prev + eps and cur[1] > nex and len(cur[0])>2): 
            morph = cur[0][-(len(cur[0])-len(prev_word)):]
            morphemes.append(morph)
            prev_word = cur[0]
    # add full word
    morph = logps[-1][0][-(len(logps[-1][0])-len(prev_word)):]
    morphemes.append(morph)
    return morphemes

# heur1: detects morpheme boundary if: 
#        (1) the current likelihood(ll) exceeds prev and next ll
def heur_prev_mid_next(logps, eps):
    morphemes = []
    prev_word = ''
    logps = [(k,v) for k,v in logps.items()]
    for i in range(1,len(logps)-1):
        if i==1:
            prev_of_prev = 0
        else:
            prev_of_prev =logps[i-2][1]
        prev = logps[i-1][1]
        cur = logps[i]
        nex = logps[i+1][1]
        if (cur[1] > prev + eps and cur[1] > nex and len(cur[0])>2): 
            morph = cur[0][-(len(cur[0])-len(prev_word)):]
            morphemes.append(morph)
            prev_word = cur[0]
    # add full word
    morph = logps[-1][0][-(len(logps[-1][0])-len(prev_word)):]
    morphemes.append(morph)
    return morphemes

def heur_reinflection(logps, eps):
    morphemes = []
    prev_word = ''
    logps = [(k,v) for k,v in logps.items()]
    for i in range(1,len(logps)-1):
        cur = logps[i]
        if len(cur[0])>2 and cur[1]==1: 
            morph = cur[0][-(len(cur[0])-len(prev_word)):]
            morphemes.append(morph)
            prev_word = cur[0]
    # add full word
    morph = logps[-1][0][-(len(logps[-1][0])-len(prev_word)):]
    morphemes.append(morph)
    return morphemes

'''# returns log likelihood of given word and its subwords
def get_logps(args, word, data, from_file=False):
    if from_file:
        with open(args.fprob, 'r') as json_file:
            logps = json.load(json_file)
            return logps[word]
    else:    
        with torch.no_grad():
            root_fhs, fcs, z = args.model.encoder(data)
            quantized_input_root, vq_loss, quantized_inds, other_inds = args.model.vq_layer_root(root_fhs,0)
            logps = dict()
            logps[word] = 1
            word_reinflections = []
            for ind in other_inds[0]:
                root_ind = ind.item()
                word_reinflections.extend(list(reinflect(args, root_ind, dict()).values()))
            # loop through word's subwords 
            for i in range(len(data[0])-2, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                logps[subword] = 0 
                #for key,val in word_reinflections.items():
                if subword+'</s>' in word_reinflections:
                    logps[subword] = 1
        logps = dict(reversed(list(logps.items())))
        return logps'''

# returns log likelihood of given word and its subwords
def get_logps_2(args, word, data, from_file=False):
    if from_file:
        with open(args.fprob, 'r') as json_file:
            logps = json.load(json_file)
            return logps[word]
    else:    
        with torch.no_grad():
            root_fhs, fcs, z = args.model.encoder(data)
            #quantized_input_root, vq_loss, quantized_inds = args.model.vq_layer_root(root_fhs,0)
            logps = dict()
            logps[word] = np.exp(args.model.log_probability_w_recon(data, root_fhs, args.recon_type))
            # loop through word's subwords 
            for i in range(len(data[0])-2, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                if args.sample_type == 'subword_given':# sample z from subword 
                    logps[subword] = np.exp(args.model.log_probability_w_recon(subdata.unsqueeze(0), None, args.recon_type))
                else:
                    logps[subword] = np.exp(args.model.log_probability_w_recon(subdata.unsqueeze(0), root_fhs, args.recon_type))
        logps = dict(reversed(list(logps.items())))
        return logps

# returns log likelihood of given word and its subwords
def get_logps(args, word, data, from_file=False):
    if from_file:
        with open(args.fprob, 'r') as json_file:
            logps = json.load(json_file)
            return logps[word]
    else:    
        with torch.no_grad():
            root_fhs, fcs, z = args.model.encoder(data)
            #quantized_input_root, vq_loss, quantized_inds = args.model.vq_layer_root(root_fhs,0)
            logps = dict()
            logps[word] = np.exp(args.model.log_probability_w_reinflection(data, root_fhs, args.recon_type))
            # loop through word's subwords 
            for i in range(len(data[0])-2, 1, -1):
                eos  = torch.tensor([2]).to(args.device)
                subdata = torch.cat([data[0][:i], eos])
                subword = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                if args.sample_type == 'subword_given':# sample z from subword 
                    logps[subword] = np.exp(args.model.log_probability_w_reinflection(subdata.unsqueeze(0), None, args.recon_type))
                else:
                    logps[subword] = np.exp(args.model.log_probability_w_reinflection(subdata.unsqueeze(0), root_fhs, args.recon_type))
        logps = dict(reversed(list(logps.items())))
        return logps

def reinflect(args, root_ind, reinflected_words):
    z0 = args.model.vq_layer_root.embedding.weight[root_ind].unsqueeze(0).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid).to(args.device)
    sft = nn.Softmax(dim=1)
    reinflected_words = dict()
    for j1 in range( args.model.orddict_emb_num):
        for j2 in range( args.model.orddict_emb_num):
            for j3 in range( args.model.orddict_emb_num):
                vq_vectors = []
                z1 = args.model.ord_vq_layers[0].embedding.weight[j1].unsqueeze(0).unsqueeze(0)
                z2 = args.model.ord_vq_layers[1].embedding.weight[j2].unsqueeze(0).unsqueeze(0)
                z3 = args.model.ord_vq_layers[2].embedding.weight[j3].unsqueeze(0).unsqueeze(0)
                vq_vectors.append(z0)
                vq_vectors.append(z1)
                vq_vectors.append(z2)
                vq_vectors.append(z3)
                vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
                vq_code = str(j1) + '-' + str(j2) + '-' + str(j3)
                root_z, suffix_z = vq_vectors
                batch_size, seq_len, _ = root_z.size()
                z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)
                root_z = root_z.permute((1,0,2))
                c_init = root_z 
                h_init = torch.tanh(c_init)
                decoder_hidden = (c_init, h_init)
                copied = []; i = 0
                while True:
                    # (1,1,ni)
                    word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0).to(args.device))
                    word_embed = torch.cat((word_embed, z_), -1)
                    # output: (1,1,dec_nh)
                    output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
                    # (1, vocab_size)
                    output_logits = args.model.decoder.pred_linear(output).squeeze(1)
                    input = torch.argmax(sft(output_logits)) 
                    char = args.vocab.id2word(input.item())
                    copied.append(char)
                    if char == '</s>':
                        reinflected_words[vq_code] = ''.join(copied)
                        break
    return reinflected_words


def config():
     # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vqvae_1x10000_8x6'
    model_path, model_vocab  = get_model_info(model_id)
    # heuristic
    args.heur_type = 'prev_mid_next'; args.eps = 0.0
    args.ll_type = 'reinflect';
    args.sample_type = 'word_given'
    # (a) avg: averages ll over word tokens, (b) sum: adds ll over word tokens
    args.recon_type = 'avg' 
    # logging
    args.logdir = 'evaluation/morph_segmentation/results/vqvae/'+model_id+'/'+args.ll_type+'/'+args.recon_type+'/'+args.sample_type+'/'+args.heur_type+'/eps'+str(args.eps)+'/'
    args.fseg   = args.logdir +'segments.txt'
    args.fprob  = args.logdir +'probs.json'
    args.load_probs_from_file = False; args.save_probs_to_file = not args.load_probs_from_file
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")
    # initialize model
    # load vocab (to initialize the model with correct vocabsize)
    with open(model_vocab) as f:
        word2id = json.load(f)
        args.vocab = VocabEntry(word2id)
    
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.ni = 256; args.enc_nh = 512;  args.dec_nh = 512; 
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.beta = 0.5
    args.embedding_dim = args.enc_nh
    args.rootdict_emb_dim = 512; args.num_dicts = 9; args.nz = 512; args.outcat=0; args.incat = 192
    args.rootdict_emb_num = 10000; args.orddict_emb_num  = 6
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.to(args.device)
    args.model.eval()
    # data
    #args.tstdata = 'evaluation/morph_segmentation/data/test.tur'
    #args.tstdata = 'evaluation/morph_segmentation/data/goldstdsample.tur'
    args.tstdata = 'evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur'
    args.maxtstsize = 3000
    args.batch_size = 1
    return args

def main():
    args = config()
    data, batches = build_data(args)
    word_probs = dict()
    fseg = open(args.fseg, 'w')
    # loop through each word 
    for data in batches:
        word = ''.join(args.vocab.decode_sentence(data[0][1:-1]))
        print(word)
        logps = get_logps(args, word, data, from_file=args.load_probs_from_file)
        word_probs[word] = logps
        
        # call segmentation heuristic 
        if args.heur_type == 'prev_mid_next':
            morphemes = heur_prev_mid_next(logps, args.eps)
        elif args.heur_type == 'prev_mid_next_and_prevnext_exceed':
            morphemes = heur_prev_mid_next_and_prevnext_exceed(logps, args.eps)
        elif args.heur_type == 'reinflection':
            morphemes = heur_reinflection(logps, args.eps)

        # write morphemes to file
        fseg.write(str(' '.join(morphemes)+'\n'))
    if args.save_probs_to_file:
        with open(args.fprob, 'w') as json_file:
            json_object = json.dumps(word_probs, indent = 4)
            json_file.write(json_object)

if __name__=="__main__":
    main()
