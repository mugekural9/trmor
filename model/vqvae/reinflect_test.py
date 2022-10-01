import torch
import torch.nn as nn
import re, torch, json, os, argparse, random
from common.utils import *
from common.vocab import VocabEntry


def oracle(args, reinflected_word):
    x = torch.tensor([args.surf_vocab.word2id['<s>']] + args.surf_vocab.encode_sentence(reinflected_word) + [args.surf_vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss(x, None, 0, 'tst')
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])
    return mapped_inds#''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds +'\t---> ' + reinflected_word


def reinflect(args, inflected_word, reinflect_tag):
    x = torch.tensor([args.surf_vocab.word2id['<s>']] + args.surf_vocab.encode_sentence(inflected_word) + [args.surf_vocab.word2id['</s>']]).unsqueeze(0).to('cuda')
    vq_vectors = []
  
    # kl version
    fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(x)
    _root_fhs = mu.unsqueeze(0)
    #_root_fhs = args.model.reparameterize(mu, logvar)
    _root_fhs = args.model.z_to_dec(_root_fhs)

    vq_vectors.append(_root_fhs)

    bosid = args.surf_vocab.word2id['<s>']
    input = torch.tensor(bosid).to('cuda')
    sft = nn.Softmax(dim=1)
    # Quantized Inputs

    vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=2:
        vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=3:
        vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=4:
        vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=5:
        vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=6:
        vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=8:
        vq_vectors.append(args.model.ord_vq_layers[6].embedding.weight[reinflect_tag[6]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[7].embedding.weight[reinflect_tag[7]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=10:
        vq_vectors.append(args.model.ord_vq_layers[8].embedding.weight[reinflect_tag[8]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[9].embedding.weight[reinflect_tag[9]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=11:
        vq_vectors.append(args.model.ord_vq_layers[10].embedding.weight[reinflect_tag[10]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=12:
        vq_vectors.append(args.model.ord_vq_layers[11].embedding.weight[reinflect_tag[11]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=16:
        vq_vectors.append(args.model.ord_vq_layers[12].embedding.weight[reinflect_tag[12]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[13].embedding.weight[reinflect_tag[13]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[14].embedding.weight[reinflect_tag[14]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[15].embedding.weight[reinflect_tag[15]].unsqueeze(0).unsqueeze(0))

    vq_vectors = (vq_vectors[0], torch.cat(vq_vectors[1:],dim=2))
    root_z, suffix_z = vq_vectors
    batch_size, seq_len, _ = fhs.size()
    z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)

    c_init = root_z
    h_init = torch.tanh(c_init)
    decoder_hidden = (h_init, c_init)
    copied = []; i = 0
    MAX_LENGTH = 50
    c=0
    while c<MAX_LENGTH:
        c+=1
        # (1,1,ni)
        word_embed = args.model.decoder.embed(torch.tensor([input]).unsqueeze(0).to('cuda'))
        word_embed = torch.cat((word_embed, z_), -1)
        # output: (1,1,dec_nh)
        output, decoder_hidden = args.model.decoder.lstm(word_embed, decoder_hidden)
        # (1, vocab_size)
        output_logits = args.model.decoder.pred_linear(output).squeeze(1)
        input = torch.argmax(sft(output_logits))
        char = args.surf_vocab.id2word(input.item())
        copied.append(char)
        if char == '</s>':
            return(''.join(copied), (reinflect_tag))
    return(''.join(copied), (reinflect_tag))



# CONFIG
def config():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'turkish_late'
    args.lang ='turkish'

    args.model_id = model_id
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.dec_nh = 512  
    args.beta = 0.2
    args.nz = 128; 
    args.num_dicts = 11
    args.outcat=0; 
    args.orddict_emb_num =  6
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)


    if 'late' in model_id:
        from model.vqvae.sig2016.early_sup.vqvae_kl_bi_early_sup import VQVAE

        model_path, model_vocab, tag_vocabs  = get_model_info(model_id, lang=args.lang)
        with open(model_vocab) as f:
            word2id = json.load(f)
            args.surf_vocab = VocabEntry(word2id)

        args.tag_vocabs = dict()
        args.enc_nh = 660
        args.incat = args.enc_nh; 
        args.emb_dim = 128#args.dec_nh
        j=0
        for tag_vocab in tag_vocabs:
            with open(tag_vocab) as f:
                word2id = json.load(f)
                args.tag_vocabs[j] = VocabEntry(word2id) 
                j+=1
        args.model = VQVAE(args, args.surf_vocab, args.tag_vocabs, model_init, emb_init, dict_assemble_type='sum_and_concat')
 

    # logging
    args.logdir = 'model/vqvae/results/tagmapping_rnn/'+args.lang+'/'+model_id+'/'
    args.logfile = args.logdir + '/copies.txt'
    args.epochs = 200
    
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")


    args.save_path = args.logdir +  str(args.epochs)+'epochs.pt'
    args.log_path =  args.logdir +  str(args.epochs)+'epochs.log'
    args.logger = Logger(args.log_path)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    args.model.to('cuda')
    return args
args = config()
model = args.model
oracle_keys = oracle(args, 'dinlediler')

key = [int(n) for n in oracle_keys.split('-')]
reinflected_word, suffix_code =  reinflect(args, 'gidecekti',key)
print(reinflected_word)
