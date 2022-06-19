# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
#from vqvae import VQVAE
from vqvae_discrete import VQVAE
from vqvae_kl import VQVAE

from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def analysis(args, asked_tag, inflected_word, reinflected_word, itr=0):

    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(reinflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_ = args.model.vq_loss(x, 0)
      
    if args.model_type =='discrete':
        quantized_inds = quantized_inds[1:]
    
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])

    return ''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.1_16x6_dec128_suffixd512'
    model_id = 'kl0.1_8x6_dec128_suffixd512'
    model_id = 'discretelemma_bilstm_8x6_dec512_suffixd512'
    model_id = 'bi_kl0.1_16x6_dec128_suffixd512'

    args.model_id = model_id
    args.num_dicts = 16
    args.orddict_emb_num  = 6
    args.lemmadict_emb_num  = 5000
    args.nz = 128; 
    args.root_linear_h = args.nz
    args.enc_nh = 512;
    args.dec_nh = args.root_linear_h
    args.incat = args.enc_nh


    if 'uni' in model_id:
        from vqvae import VQVAE
        args.model_type = 'unilstm'

    if 'kl' in model_id:
        from vqvae_kl import VQVAE
        args.model_type = 'kl'
        args.dec_nh = args.nz  

    if 'bi' in model_id:
        from vqvae_bidirect import VQVAE
        args.model_type = 'bilstm'

    if 'discrete' in model_id:
        from vqvae_discrete import VQVAE
        args.num_dicts = 9
        args.dec_nh = args.enc_nh
        args.model_type = 'discrete'

    if 'bi_kl' in model_id:
        from vqvae_kl_bi import VQVAE
        args.model_type = 'bi_kl'
        args.dec_nh = args.nz  

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

    args.embedding_dim = args.enc_nh; #args.nz = args.enc_nh
    args.beta = 0.5
    args.outcat=0; args.incat = 512
    if args.model_type == 'bi_kl':
        args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)
    else:
        args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')


    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    with open('data/sigmorphon2016/turkish-task3-train', 'r') as reader:
        with open(args.logdir+'train_'+args.model_id+'_shuffled.txt', 'w') as writer:
            # create datasets   
            lines = []
            for line in reader:
                split_line = line.split('\t')
                asked_tag  = split_line[1]
                inflected_word = split_line[0]
                reinflected_word = split_line[2].strip()
                lines.append(analysis(args, asked_tag, inflected_word, reinflected_word)+'\n')
            random.shuffle(lines)
            for line in lines[:13000]:
                writer.write(line)
    
    with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
        with open(args.logdir+'test_'+args.model_id+'_shuffled.txt', 'w') as writer:
            # create datasets   
            lines = []
            for line in reader:
                split_line = line.split('\t')
                asked_tag  = split_line[1]
                inflected_word = split_line[0]
                reinflected_word = split_line[2].strip()
                lines.append(analysis(args, asked_tag, inflected_word, reinflected_word)+'\n')
            random.shuffle(lines)
            for line in lines[:13000]:
                writer.write(line)
            


if __name__=="__main__":
    main()

