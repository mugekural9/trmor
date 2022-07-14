# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
#from vqvae import VQVAE
from vqvae_discrete import VQVAE
from vqvae_kl_bi import VQVAE

from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
from collections import defaultdict

def analysis(args, asked_tag, inflected_word, reinflected_word, itr=0):

    x = torch.tensor([args.surf_vocab.word2id['<s>']] + args.surf_vocab.encode_sentence(reinflected_word) + [args.surf_vocab.word2id['</s>']]).unsqueeze(0)
    if 'supervision' in args.model_id:
        quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss(x, 0, mode='test')
    #quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss_test(x, 0, mode='test')
      
    #if args.model_type =='discrete' or args.model_type == 'discrete_sum':
    #    quantized_inds = quantized_inds[1:]
    
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])

    return ''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds + '\t'+ reinflected_word



def config(model_id, num_dicts, orddict_emb_num, lang):
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    args.lang = lang

    args.model_id = model_id
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.dec_nh = 256  
    args.beta = 0.2
    args.nz = 128; 
    args.num_dicts = num_dicts
    args.outcat=0; 
    args.orddict_emb_num =  orddict_emb_num
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)


    if 'supervision' in model_id:
        from model.vqvae.vqvae_kl_bi_early_sup import VQVAE
        from model.vqvae.vqvae_kl_bi_late_sup import VQVAE

        model_path, model_vocab, tag_vocabs  = get_model_info(model_id, lang=args.lang)
        with open(model_vocab) as f:
            word2id = json.load(f)
            args.surf_vocab = VocabEntry(word2id)

        args.tag_vocabs = dict()
        args.enc_nh = 660
        args.incat = args.enc_nh; 
        j=0
        for tag_vocab in tag_vocabs:
            with open(tag_vocab) as f:
                word2id = json.load(f)
                args.tag_vocabs[j] = VocabEntry(word2id) 
                j+=1
        args.model = VQVAE(args, args.surf_vocab, args.tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat')

    elif 'bi_kl' in model_id:
        from vqvae_kl_bi import VQVAE
        args.model_type = 'bi_kl'
        args.enc_nh = 300
        args.incat = args.enc_nh; 
        args.emb_dim = 256
        model_path, model_vocab  = get_model_info(model_id, lang=args.lang)
        with open(model_vocab) as f:
            word2id = json.load(f)
            args.surf_vocab = VocabEntry(word2id)
        args.model = VQVAE(args, args.surf_vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

    # logging
    args.logdir = 'model/vqvae/results/analysis/'+args.lang+'/'+model_id+'/'
    args.logfile = args.logdir + '/copies.txt'
    
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args




'''def config(model_id, num_dicts, orddict_emb_num, lang):
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    args.model_id = model_id
    args.lemmadict_emb_num  = 5000
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.enc_nh = 300
    args.dec_nh = 256  
    args.embedding_dim = args.enc_nh
    args.beta = 0.2
    args.nz = 128; 
    args.num_dicts = num_dicts
    args.outcat=0; 
    args.orddict_emb_num =  orddict_emb_num
    args.incat = args.enc_nh; 
    args.lang = lang

    if 'discrete' in model_id:
        from vqvae_discrete import VQVAE
        args.num_dicts = 9
        args.dec_nh = args.enc_nh
        args.model_type = 'discrete'

    if 'bi_kl' in model_id:
        from vqvae_kl_bi import VQVAE
        args.model_type = 'bi_kl'

    if 'discrete' in model_id and 'sum' in model_id:
        from vqvae_discrete_sum import VQVAE
        args.num_dicts = 5#9
        args.dec_nh = args.enc_nh
        args.incat = int(args.enc_nh/(args.num_dicts-1))        
        args.model_type = 'discrete_sum'

    model_path, model_vocab  = get_model_info(model_id, lang=args.lang)
    # logging
    args.logdir = 'model/vqvae/results/analysis/'+args.lang+'/'+model_id+'/'
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
        args.surf_vocab = VocabEntry(word2id)
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.beta = 0.5
    args.model = VQVAE(args, args.surf_vocab, model_init, emb_init, dict_assemble_type='sum_and_concat', bidirectional=True)
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args'''


def counter(model_id, lang, logger):
    dict_combs = defaultdict(lambda: 0)
    unseen_combs = defaultdict(lambda: 0)
    with open('model/vqvae/results/analysis/'+lang+'/'+model_id+'/train_'+model_id+'_shuffled.txt','r') as reader:
        for line in reader:
            dict_combs[line.split('\t')[-2]] +=1
    with open('model/vqvae/results/analysis/'+lang+'/'+model_id+'/test_'+model_id+'_shuffled.txt','r') as reader:
        testsize = 0
        for line in reader:
            testsize +=1
            comb= line.split('\t')[-2]
            if comb not in dict_combs:
                unseen_combs[comb] +=1
    logger.write('\nmodel: %s' % model_id)
    logger.write('\n%d combs never seen' % len(unseen_combs))
    logger.write('\n%d totally in test data over %d' % (sum(unseen_combs.values()), testsize))
    logger.write('\nratio: %.3f\n' % (sum(unseen_combs.values())/testsize))


def tag_analysis(model_id, num_dicts, orddict_emb_num, lang, logger, during_training=False, args=None):
    if not during_training:
        args = config(model_id, num_dicts, orddict_emb_num, lang)
    else:
        args.logdir = 'model/vqvae/results/analysis/'+args.lang+'/'+model_id+'/'
        try:
            os.makedirs(args.logdir)
            print("Directory " , args.logdir ,  " Created ") 
        except FileExistsError:
            print("Directory " , args.logdir ,  " already exists")

    with open('data/sigmorphon2016/'+args.lang+'-task3-train', 'r') as reader:
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
            for line in lines:
                writer.write(line)
    with open('data/sigmorphon2016/'+args.lang+'-task3-test', 'r') as reader:
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
            for line in lines:
                writer.write(line)
   
            


if __name__=="__main__":
    tag_analysis()

