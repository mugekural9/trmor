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

def copy(args, word, itr=0):
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
            return ''.join(copied), quantized_inds, vq_loss

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vqvae_segm'
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
    args.ni = 256; 
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.2; args.dec_dropout_out = 0.3
    args.enc_nh = 512;
    args.dec_nh = args.enc_nh; args.embedding_dim = args.enc_nh; #args.nz = args.enc_nh
    args.beta = 0.5
    args.rootdict_emb_dim = 512; args.num_dicts = 3; args.nz = 512; args.outcat=0; args.incat = 512
    args.rootdict_emb_num = 8000; args.orddict_emb_num  = 1280; args.orddict_emb_num_2  = 1280
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')


    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    # copy tst data
    #args.tstdata = 'model/vqvae/data/surf.uniquesurfs.val.txt'
    args.tstdata  = 'model/vqvae/data/theval_disrupted.tur'
    args.maxtstsize = 1000
    tstdata = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')   
    different_words = dict() 
    with open(args.logfile, "w") as f:
        for itr,data in enumerate(tstdata):
            if itr>=20:
                break
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            #copied_word, selected_inds = copy(args, word, itr)
            #if copied_word not in different_words:
            #    different_words[copied_word] = 1
            #else:
            #    different_words[copied_word] += 1
            #f.write(copied_word + "\t"+ str(selected_inds)+ "\n")
            root_id = copy(args, word)[1][0][0][0].item()
            # copy word
            for i in range(1,len(word)+1):
                copied_word, inds, vq_loss = copy(args, word[:i])
                subword_root_id = inds[0][0][0].item()
                #if subword_root_id == root_id and word[:i]+'</s>' == copied_word:
                if True:#word[:i]+'</s>' == copied_word:
                    print(word[:i],' --->', copied_word)
                    print(inds)
                    print('losses:', vq_loss)
                    print('\n')

if __name__=="__main__":
    main()



