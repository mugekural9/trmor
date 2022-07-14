# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from msved import MSVED

from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def analysis(args, asked_tag, inflected_word, reinflected_word, itr=0):

    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(reinflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
    loss_ux_msvae, ux_msvae_recon_loss, ux_msvae_kl_loss, ux_msvae_recon_acc, gumbel_logits = args.model.loss_ux_msvae(x, 0.01, 0.94)
    mapped_inds =  '-'.join([str(i.tolist()) for i in gumbel_logits])

    return ''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds+'\t'+ reinflected_word

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'msved_unlabeled'
    args.model_id = model_id

    # training
    args.batchsize = 128; args.epochs = 175
    args.opt= 'Adam'; args.lr = 0.001
    args.task = 'msved'
    args.seq_to_no_pad = 'surface'
    # data
    args.trndata  = 'data/sigmorphon2016/turkish-task3-train'
    args.valdata  = 'data/sigmorphon2016/turkish-task3-test'
    args.tstdata  = 'data/sigmorphon2016/turkish-task3-test'
    args.unlabeled_data = 'data/sigmorphon2016/zhou_merged'
    args.surface_vocab_file = args.trndata
    args.maxtrnsize = 700000000; args.maxvalsize = 10000; args.maxtstsize = 10000
    # model
    args.mname = 'msved' 
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    args.ni = 300; args.nz = 150; 
    args.enc_nh = 256; args.dec_nh = 256
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; 

    model_path, model_vocab, tag_vocabs  = get_model_info(model_id)
    # initialize model
    # load vocab (to initialize the model with correct vocabsize)
    with open(model_vocab) as f:
        word2id = json.load(f)
        args.vocab = VocabEntry(word2id)
    tag_vocabs_dict = dict()
    keys = [
        ['case'], 
        ['polar'],
        ['mood'],
        ['evid'],
        ['pos'],
        ['per'],['num'],
        ['tense'],
        ['aspect'],
        ['inter'],
        ['poss']]
    
    for i, tfile in enumerate(tag_vocabs):
        with open(tfile) as f:
            word2id = json.load(f)
            print(word2id)
            print(keys[i][0])
            tag_vocabs_dict[keys[i][0]] = VocabEntry(word2id)
            
    
    args.logdir = 'model/msved/results/tagmapping_rnn_vq_tags/'+model_id+'/'
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")
    args.model = MSVED(args, args.vocab, tag_vocabs_dict, model_init, emb_init)
    args.model.to(args.device)

    args.save_path = args.logdir +  str(args.epochs)+'epochs.pt'
    args.log_path =  args.logdir +  str(args.epochs)+'epochs.log'
    args.logger = Logger(args.log_path)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    args.model.to('cuda')
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
            #random.shuffle(lines)
            for line in lines[:13000]:
                writer.write(line)
            


if __name__=="__main__":
    main()

