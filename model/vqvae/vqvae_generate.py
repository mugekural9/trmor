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
#    j,k,m = itr
    (j,k,m,l,o,h) = itr
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    fhs, _ ,_  =  args.model.encoder(x)
    vq_vectors = []; vq_losses = []; vq_inds = []
    # quantized_input_root: (B, 1, hdim)
    quantized_input_root, vq_loss, quantized_inds_root = args.model.vq_layer_root(fhs,0, forceid=774)
    vq_inds.append(quantized_inds_root.item())
    vq_vectors.append(quantized_input_root)
    _fhs =  args.model.linear_suffix(fhs)
    quantized_input_suffix, vq_loss, quantized_inds_suffix = args.model.vq_layer_suffix(_fhs,0, forceid= j)
    vq_inds.append(quantized_inds_suffix.item())
    vq_vectors.append(quantized_input_suffix)
    i=1
    # quantize thru ord dicts
    for linear, vq_layer in zip(args.model.ord_linears, args.model.ord_vq_layers):
    #for vq_layer in self.ord_vq_layers:
        _fhs =  linear(fhs)
        # quantized_input: (B, 1, orddict_emb_dim_2)
        quantized_input, vq_loss, quantized_inds = vq_layer(_fhs, 0, forceid= itr[i])
        vq_vectors.append(quantized_input)
        vq_inds.append(quantized_inds.item())
        i+=1
    vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2)) 
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
        word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0))
        word_embed = torch.cat((word_embed, z_), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = args.model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits)) 
        char = args.vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            #print(''.join(copied))
            return ''.join(copied), vq_inds

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vqvae_6x8'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/generation/'+model_id+'/'
    args.logfile = args.logdir + '/kesfet_samples_6x8.txt'
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
    args.dec_nh = args.enc_nh; args.embedding_dim = args.enc_nh; 
    args.beta = 0.25
    args.rootdict_emb_dim = 512
    args.rootdict_emb_num = 1000
    args.orddict_emb_num  = 8
    args.rootdict_emb_dim = 512;  args.nz = 512; 
    args.num_dicts = 7; args.outcat=0; args.incat = 192
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')
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
            if itr>=1:
                break
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            for j in range(8):
                for k in range(8):
                    for m in range(8):
                        for l in range(8):
                            for o in range(8):
                                for h in range(8):
                                    copied_word, selected_inds = copy(args, word, (j,k,m,l,o,h))
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



