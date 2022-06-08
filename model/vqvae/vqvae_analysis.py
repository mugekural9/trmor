# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vqvae import VQVAE
from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def copy(args, asked_tag, inflected_word, reinflected_word, itr=0):

    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(reinflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_ = args.model.vq_loss(x, 0)
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])
    return ''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = '2x100dec512_suffixd512'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/analysis/'+model_id+'/'
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
    args.ni = 256; 
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.enc_nh = 512;
    args.dec_nh = 512#args.enc_nh; 
    args.embedding_dim = args.enc_nh; #args.nz = args.enc_nh
    args.beta = 0.5
    args.num_dicts = 2; args.nz = 512; args.outcat=0; args.incat = 512
    args.orddict_emb_num  = 100; 
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')


    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
        with open('tst_2x100_512.txt', 'w') as writer:
            for line in reader:
                split_line = line.split('\t')
                asked_tag  = split_line[1]
                inflected_word = split_line[0]
                reinflected_word = split_line[2].strip()
                writer.write(copy(args, asked_tag, inflected_word, reinflected_word)+'\n')

if __name__=="__main__":
    main()



