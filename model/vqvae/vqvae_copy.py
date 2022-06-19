# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def copy(args, word, itr=0):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _, _, _ = args.model.vq_loss(x, 0)
    ##v4
    root_z, suffix_z = quantized_inputs
    c_init = root_z
    h_init = torch.tanh(c_init)
    decoder_hidden = (h_init, c_init)
    copied = []; 
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
            return ''.join(copied), [t.item() for t in quantized_inds]#, vq_loss

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'unilstm_4x10_dec100_suffixd512'
    model_id = 'discrete_4x10_dec100_suffixd512'
    model_id = 'kl0.1_4x10_dec128_suffixd512'
    model_id = 'bilstm_4x10_dec100_suffixd512'
    model_id = 'kl0.1_8x10_dec150_suffixd512'
    model_id = 'kl0.1_4x10_dec128_suffixd512'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.1_16x10_dec128_suffixd512'
    model_id = 'kl0.1_8x6_dec128_suffixd512'

    args.num_dicts = 8
    args.orddict_emb_num  = 6
    args.lemmadict_emb_num  = 5000
    args.nz = 128
    args.root_linear_h = args.nz
    args.dec_nh= args.root_linear_h
    args.enc_nh = 512    
    args.incat = args.enc_nh 

    if 'uni' in model_id:
        from vqvae import VQVAE
        args.model_type = 'unilstm'

    if 'kl' in model_id:
        from vqvae_kl import VQVAE
        args.model_type = 'kl'
        args.nz = args.root_linear_h
        args.dec_nh = args.nz  
        

    if 'bi' in model_id:
        from vqvae_bidirect import VQVAE
        args.model_type = 'bilstm'

    if 'discrete' in model_id:
        from vqvae_discrete import VQVAE
        args.model_type = 'discrete'

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
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.embedding_dim = args.enc_nh; #args.nz = args.enc_nh
    args.beta = 0.5
    args.outcat=0; 
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    # copy tst data
    args.tstdata  = 'data/sigmorphon2016/turkish-task3-dev'
    args.maxtstsize = 1000
    tstdata = read_data(args.maxtstsize, args.tstdata, args.vocab, 'TST')   
    true = 0; false = 0
    with open(args.logfile, "w") as f:
        for itr,data in enumerate(tstdata):
            word =  ''.join(args.vocab.decode_sentence_2(data[0]))
            copied_word, selected_inds = copy(args, word, itr)
            copied_word= copied_word[:-4]
            f.write(word + "\t--->"+ copied_word+ '\t'+ str(selected_inds)+"\n")
            if copied_word != word:
                false+=1
                print(word + "\t--->"+ copied_word+ '\t'+ str(selected_inds))
            else:
                true +=1
        print('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/len(tstdata)))
        f.write('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/(true+false)))

if __name__=="__main__":
    main()



