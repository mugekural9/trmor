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

def copy(args, word, itr):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    fhs, _ ,_  =  args.model.encoder(x)
    # quantized_inputs: (B, 1, hdim)
    #fhs_1 = args.model.linear_1(fhs)
    quantized_inputs, vq_loss, quantized_inds, rates, dist_sum = args.model.vq_layer(fhs,0, forceid=681)

    fhs_2 = args.model.linear_2(fhs)
    quantized_inputs_2, vq_loss_2, quantized_inds_2, _, _ = args.model.vq_layer_2(fhs_2,0, forceid=itr)

    fhs_3 = args.model.linear_3(fhs)
    quantized_inputs_3, vq_loss_3, quantized_inds_3, _, _ = args.model.vq_layer_3(fhs_3,0, forceid=itr+1)

    fhs_4 = args.model.linear_4(fhs)
    quantized_inputs_4, vq_loss_4, quantized_inds_4, _, _ = args.model.vq_layer_4(fhs_4,0, forceid=0)

    quantized_inputs += quantized_inputs_2 + quantized_inputs_3 + quantized_inputs_4

    selected_inds = (quantized_inds.item(), quantized_inds_2.item(), quantized_inds_3.item(), quantized_inds_4.item())

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
            return ''.join(copied), selected_inds
            break

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'mvqvae_003'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/copying/'+model_id+'/'
    args.logfile = args.logdir + '/oku_copies.txt'
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
    args.dec_nh = args.enc_nh; args.embedding_dim = args.enc_nh; args.nz =  args.enc_nh 
    args.num_embeddings = 710
    args.beta = 0.25
    args.model = VQVAE(args, args.vocab, model_init, emb_init)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    # copy tst data
    #args.tstdata = 'model/vqvae/data/surf.uniquesurfs.val.txt'
    args.tstdata = 'model/vqvae/data/sosimple.new.seenroots.val.txt'
    args.maxtstsize = 1000
    tstdata = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')   
    different_words = dict() 
    with open(args.logfile, "w") as f:
        for itr,data in enumerate(tstdata):
            if itr>=5:
                break
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            copied_word, selected_inds = copy(args, word, itr)
            if copied_word not in different_words:
                different_words[copied_word] = 1
            else:
                different_words[copied_word] += 1
            f.write(copied_word + "\t"+ str(selected_inds)+ "\n")
    '''# copy word
    word = "CIkIlmayacaktI"
    for i in range(len(word)):
        print('%s --> %s' % (word[:i+1],copy(model, word[:i+1])))'''

if __name__=="__main__":
    main()



