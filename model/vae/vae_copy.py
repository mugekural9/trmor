# -----------------------------------------------------------
# Date:        2021/12/26
# Author:      Muge Kural
# Description: Word copier for trained VAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vae import VAE
from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def copy(args, word):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    nsamples = 1
    mu, logvar, _ = args.model.encoder(x)
    # (1,1,nz)
    #z = model.reparameterize(mu, logvar, nsamples)
    z = mu.unsqueeze(0)
    # (1,1,dec_nh)
    c_init = args.model.decoder.trans_linear(z)
    h_init = torch.tanh(c_init)
    decoder_hidden = (c_init, h_init)
    copied = []; i = 0
    while True:
        # (1,1,ni)
        word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0))
        word_embed = torch.cat((word_embed, z), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = args.model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits)) 
        char = args.vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            #print(''.join(copied))
            return ''.join(copied)
            break

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vae_segm_06_40'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vae/results/copying/'+model_id+'/'
    args.logfile = args.logdir + '/copies.txt'
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
    # copy tst data
    args.tstdata = 'data/unlabelled/theval.tur'
    args.maxtstsize = 10000
    tstdata = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')    
    with open(args.logfile, "w") as f:
        for data in tstdata:
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            f.write(copy(args, word) + "\n")

    '''# copy word
    word = "CIkIlmayacaktI"
    for i in range(len(word)):
        print('%s --> %s' % (word[:i+1],copy(model, word[:i+1])))'''

if __name__=="__main__":
    main()



