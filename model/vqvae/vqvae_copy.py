# -----------------------------------------------------------
# Date:        2021/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vqvae import VQVAE
from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def copy(args, word):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    fhs =  args.model.encoder(x)
    # quantized_inputs: (B, 1, hdim)
    quantized_inputs, vq_loss = args.model.vq_layer(fhs)
    z = quantized_inputs
    c_init = z #args.model.decoder.trans_linear(z)
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
    model_id = 'vqvae_1'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/copying/'+model_id+'/'
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
    args.ni = 512; 
    args.enc_nh = 1024; args.dec_nh = 1024
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.3; args.dec_dropout_out = 0.5
    args.embedding_dim = args.enc_nh
    args.nz = args.embedding_dim 
    args.num_embeddings = 10
    args.beta = 0.25
    args.model = VQVAE(args, args.vocab, model_init, emb_init)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    # copy tst data
    args.tstdata = 'model/vqvae/data/surf.uniquesurfs.val.txt'
    args.maxtstsize = 10000
    tstdata = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')   
    different_words = dict() 
    with open(args.logfile, "w") as f:
        for data in tstdata:
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            copied_word = copy(args, word)
            if copied_word not in different_words:
                different_words[copied_word] = 1
            else:
                different_words[copied_word] += 1
            f.write(copied_word + "\n")
    breakpoint()
    '''# copy word
    word = "CIkIlmayacaktI"
    for i in range(len(word)):
        print('%s --> %s' % (word[:i+1],copy(model, word[:i+1])))'''

if __name__=="__main__":
    main()



