# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from ast import arg
from unicodedata import bidirectional
from common.vocab import VocabEntry


from data.data import read_data
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os

def reinflect(args, inflected_word, reinflect_tag):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(inflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    vq_vectors = []
  
    if args.model_type =='unilstm':
        #UNIDIRECT
        fhs, _, _ = args.model.encoder(x)
        vq_vectors.append(args.model.linear_root(fhs))

    if args.model_type =='kl':
        #KL VERSION
        fhs, _, _, mu, logvar = args.model.encoder(x)
        _root_fhs = mu.unsqueeze(0)
        #_root_fhs = args.model.reparameterize(mu, logvar)
        vq_vectors.append(_root_fhs)

    if args.model_type == 'bi_kl' or args.model_type == 'bi_kl_sum':
        # kl version
        fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(x)
        _root_fhs = mu.unsqueeze(0)
        #_root_fhs = args.model.reparameterize(mu, logvar)
        _root_fhs = args.model.z_to_dec(_root_fhs)

        vq_vectors.append(_root_fhs)

    if args.model_type =='bilstm':
        #BIDIRECT
        fhs, _, _,fwd,bck = args.model.encoder(x)   
        vq_vectors.append(args.model.linear_root(fwd))

    if args.model_type =='discrete' or args.model_type == 'discrete_sum':
        #DISCRETE
        fhs, _, _, fwd,bck = args.model.encoder(x)    
        quantized_input, vq_loss, quantized_inds = args.model.vq_layer_lemma(fwd,0)
        vq_vectors.append(quantized_input)
            

    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    # Quantized Inputs
    vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=2:
        vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=4:
        vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=5:
        vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=6:
        vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))

    if args.num_dicts >=8:
        vq_vectors.append(args.model.ord_vq_layers[6].embedding.weight[reinflect_tag[6]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[7].embedding.weight[reinflect_tag[7]].unsqueeze(0).unsqueeze(0))

    if args.num_dicts >=16:
        vq_vectors.append(args.model.ord_vq_layers[8].embedding.weight[reinflect_tag[8]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[9].embedding.weight[reinflect_tag[9]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[10].embedding.weight[reinflect_tag[10]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[11].embedding.weight[reinflect_tag[11]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[12].embedding.weight[reinflect_tag[12]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[13].embedding.weight[reinflect_tag[13]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[14].embedding.weight[reinflect_tag[14]].unsqueeze(0).unsqueeze(0))
        vq_vectors.append(args.model.ord_vq_layers[15].embedding.weight[reinflect_tag[15]].unsqueeze(0).unsqueeze(0))


    if args.model_type =='discrete_sum' or args.model_type =='bi_kl_sum':
        vq_vectors = (vq_vectors[0], torch.sum(torch.stack(vq_vectors[1:]),dim=0))
    else:
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
            return(''.join(copied), (reinflect_tag))
    return(''.join(copied), (reinflect_tag))


def oracle(args, asked_tag, inflected_word, reinflected_word, itr=0):
    #breakpoint()
    #mapped_inds = entry_dict['<s>'+reinflected_word+'</s>']
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(reinflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    quantized_inputs, vq_loss, quantized_inds, encoder_fhs, _,_,_, _ = args.model.vq_loss(x, 0, 'tst')

    if args.model_type =='discrete' or args.model_type =='discrete_sum':
        quantized_inds = quantized_inds[1:]
    mapped_inds =  '-'.join([str(i[0][0].item()) for i in quantized_inds])

    return ''.join(inflected_word) +'\t'+ asked_tag +'\t'+ mapped_inds +'\t---> ' + reinflected_word

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'unilstm_4x10_dec100_suffixd512'
    model_id = 'bilstm_4x10dec100_suffixd512'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.2_4x10_dec128_suffixd512'
    model_id = 'kl0.1_4x10_dec128_suffixd512'
    model_id = 'kl0.1_8x10_dec150_suffixd512'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.1_16x6_dec128_suffixd512'
    model_id = 'kl0.1_8x10_dec128_suffixd512'
    model_id = 'kl0.1_8x6_dec128_suffixd512'
    model_id = 'bi_kl0.2_8x6_dec128_suffixd512'
    model_id = 'bi_kl0.1_8x4_dec128_suffixd512'
    model_id = 'discretelemma_sum_bilstm_8x6_dec512_suffixd64'
    model_id = 'bi_kl0.1_sum_8x6_dec128_suffixd64'
    model_id = 'bi_kl0.1_8x12_dec128_suffixd512'
    model_id = 'bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'bi_kl0.1_16x6_dec128_suffixd512'
    model_id = 'discretelemma_bilstm_8x6_dec512_suffixd512'
    model_id = 'beta0.2_discretelemma_bilstm_8x6_dec512_suffixd512'
    model_id = 'beta0.2_bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'beta0.5_anneal_bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'REALbeta1.0_bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'REALbeta1.0_bi_kl0.1_4x6_dec128_suffixd512'
    model_id = 'REALbeta1.0_bi_kl0.1_8x2_dec128_suffixd512'
    model_id = 'discretelemma_sum_bilstm_4x6_dec512_suffixd128'
    model_id = 'REALbeta1.0_bi_kl0.1_6x8_dec128_suffixd300'
    model_id = 'REALbeta1.0_bi_kl0.1_6x6_dec128_suffixd300'
    model_id = 'lemma64_bi_kl0.1_6x6_dec64_suffixd300'
    model_id = 'logdetmax_bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'new_bi_kl0.1_4x8_dec128_suffixd512'
    model_id = 'FINAL_bi_kl0.2_5x5_dec256_suffixd300'
    args.num_dicts = 5
    
    #if str(8) in model_id:
    #   args.num_dicts = 6
    #if str(4) in model_id:
    #    args.num_dicts = 4
        
    args.lemmadict_emb_num  = 10000

    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.4; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.enc_nh = 300
    args.dec_nh = 256  
    args.embedding_dim = args.enc_nh
    args.beta = 0.5
    args.nz = 128; 
    args.num_dicts = 5; 
    args.outcat=0; 
    args.orddict_emb_num =  5
    args.incat = args.enc_nh; 


    if 'discrete' in model_id:
        from vqvae_discrete import VQVAE
        args.num_dicts = 9
        args.dec_nh = args.enc_nh
        args.model_type = 'discrete'

    if 'bi_kl' in model_id:
        from vqvae_kl_bi import VQVAE
        args.model_type = 'bi_kl'
        #args.dec_nh = args.nz  

    if 'discrete' in model_id and 'sum' in model_id:
        from vqvae_discrete_sum import VQVAE
        args.num_dicts = 5#9
        args.dec_nh = args.enc_nh
        args.incat = int(args.enc_nh/(args.num_dicts-1))        
        args.model_type = 'discrete_sum'

    if 'bi_kl' in model_id and 'sum' in model_id:
        from vqvae_kl_bi_sum import VQVAE
        args.incat = int(args.enc_nh/args.num_dicts)        
        args.model_type = 'bi_kl_sum'

    model_path, model_vocab  = get_model_info(model_id)
    args.model_id = model_id
    # logging
    args.logdir = 'model/vqvae/results/oracle/'+model_id+'/'
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
    args.beta = 0.5
    args.outcat=0;
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
    with torch.no_grad():
        #r_entries = dict()
        #with open('model/vqvae/results/training/28794_instances/'+args.model_id+'/9_cluster.json') as f:
        #    entries = json.load(f)
        #    for key,vals in entries.items():
        #        for v in vals:
        #            r_entries[v] = key[1:]
            


        with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
            with open(args.logdir + 'oracle.txt', 'w') as writer:
                for line in reader:
                    split_line = line.split('\t')
                    asked_tag  = split_line[1]
                    inflected_word = split_line[0]
                    reinflected_word = split_line[2].strip()
                    writer.write(oracle(args, asked_tag, inflected_word, reinflected_word)+'\n')

        with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected', 'w') as writer:
            with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_true', 'w') as writer_true:
                with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_false', 'w') as writer_false:
                    keys =[]
                    with open(args.logdir + 'oracle.txt', 'r') as reader:
                        for line in reader:
                            keys.append( line.split('\t')[2].strip().split('-'))

                    with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
                        true=0; false = 0
                        for i,line in enumerate(reader):
                            inflected_word, tag_name, gold_reinflection = line.strip().split('\t')
                            inflected_word = inflected_word.strip()
                            tag_name = tag_name.strip()
                            gold_reinflection = gold_reinflection.strip()
                            key = [int(n) for n in keys[i]]

                            #*******    
                            reinflected_word, vq_code =  reinflect(args, inflected_word,key)
                            reinflected_word = reinflected_word[:-4]
                            writer.write(inflected_word +'\t'+ tag_name+'\t'+reinflected_word + '\n')
                            if reinflected_word == gold_reinflection:
                                true +=1
                                writer_true.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ '-'.join([str(s) for s in vq_code])+'\n')
                            else:
                                false+=1
                                writer_false.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ '-'.join([str(s) for s in vq_code])+'\n')

                    print('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/(true+false)))
                    writer_true.write('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/(true+false)))

if __name__=="__main__":
    main()



