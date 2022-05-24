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

def reinflect(args, root_ind, reinflected_words):
    z0 = args.model.vq_layer_root.embedding.weight[root_ind].unsqueeze(0).unsqueeze(0)
    probs = [];  vq_inds = []; 
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    reinflected_words[root_ind] = dict()

    for j1 in range( args.model.orddict_emb_num):
        for j2 in range( args.model.orddict_emb_num):
            for j3 in range( args.model.orddict_emb_num):
                vq_vectors = []
                z1 = args.model.ord_vq_layers[0].embedding.weight[j1].unsqueeze(0).unsqueeze(0)
                z2 = args.model.ord_vq_layers[1].embedding.weight[j2].unsqueeze(0).unsqueeze(0)
                z3 = args.model.ord_vq_layers[2].embedding.weight[j3].unsqueeze(0).unsqueeze(0)
                vq_vectors.append(z0)
                vq_vectors.append(z1)
                vq_vectors.append(z2)
                vq_vectors.append(z3)
                vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
                vq_code = str(j1) + '-' + str(j2) + '-' + str(j3)

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
                        reinflected_words[root_ind][vq_code] = ''.join(copied)
                        break

    return reinflected_words

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vqvae_1x10000_3x8'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/vqvae/results/reinflection/'+model_id+'/'
    args.logfile = args.logdir + '/'
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
    args.beta = 0.5
    args.rootdict_emb_dim = 512
    args.rootdict_emb_num = 10000
    args.orddict_emb_num  = 8
    args.rootdict_emb_dim = 512;  args.nz = 512; 
    args.num_dicts = 4; args.outcat=0; args.incat = 192
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args


def main():
    args = config()
    different_words = dict(); reinflected_words = dict()
    for i in range(10000):
        print(i)
        reinflected_words = reinflect(args, i, reinflected_words)
        with open(args.logfile+str(i)+'.json', "w") as f:
            f.write(json.dumps(reinflected_words[i]))





if __name__=="__main__":
    main()



