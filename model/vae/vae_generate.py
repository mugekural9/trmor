# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Word generator through sampling from prior for trained VAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vae import VAE
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
import numpy as np

def sample(args, z):
    # z: (1,1,nz)
    z = z.unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    # (1,1,dec_nh)
    c_init = args.model.decoder.trans_linear(z)
    h_init = torch.tanh(c_init)
    decoder_hidden = (c_init, h_init)
    sampled = []; i = 0; max_length = 50
    word = ''
    while i < max_length:
        i +=1
        # (1,1,ni)
        word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0))
        word_embed = torch.cat((word_embed, z), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = args.model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits)) 
        #input = torch.multinomial(sft(output_logits), num_samples=1) # sample
        char = args.vocab.id2word(input.item())
        sampled.append(char)
        if char == '</s>':
            word = ''.join(sampled)
            print(word)
            return word
            #break

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vae_segm_06_40'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vae/results/generation/'+model_id+'/'
    args.logfile = args.logdir + '/samples.txt'
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
    args.ni = 256; args.nz = 32; 
    args.enc_nh = 256; args.dec_nh = 256
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.model = VAE(args, args.vocab, model_init, emb_init)
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    num_samples = 100
    torch.cuda.manual_seed(1)
    z = args.model.prior.sample((num_samples,)).to('cpu:0').unsqueeze(0)
    # generate word
    with open(args.logfile, "w") as f:
        # interpolate
        lineno1 = 5
        lineno2 = 8
        z1 = z[:,lineno1-1,:]
        z2 = z[:,lineno2-1,:]
        n= 10
        z3 = []
        for i in range(n):
            cof = 1.0*i/(n-1)
            print(cof)
            zi = torch.lerp(z1, z2, cof)
            z3.append(np.expand_dims(zi.cpu(), axis=0))
        z3 = torch.tensor(np.concatenate(z3, axis=0)).permute(1,0,2)
        z = torch.cat([z,z3],1)
        for i in range(num_samples+n):
            word = sample(args, z[:,i,:])
            f.write(word + "\n")

if __name__=="__main__":
    main()



