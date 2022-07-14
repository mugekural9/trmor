# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from common.utils import *
import argparse, torch, json,  os
from collections import defaultdict


def reinflect(args, inflected_word, reinflect_tag):
    
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(inflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    vq_vectors = []


    if args.model_type == 'bi_kl' or args.model_type == 'bi_kl_sum':
        # kl version
        fhs, _, _, mu, logvar, fwd,bck = args.model.encoder(x)
        _root_fhs = mu.unsqueeze(0)
        #_root_fhs = args.model.reparameterize(mu, logvar)
        _root_fhs = args.model.z_to_dec(_root_fhs)
        vq_vectors.append(_root_fhs)

    if args.model_type =='discrete' or args.model_type == 'discrete_sum':
        #DISCRETE
        fhs, _, _, fwd,bck = args.model.encoder(x)    
        quantized_input, vq_loss, quantized_inds = args.model.vq_layer_lemma(fwd,0)
        vq_vectors.append(quantized_input)

    new_vq_vectors = [ [], [], [] ,[] ,[],[],[],[] ]  
    for k,v in reinflect_tag.items():
      for e,en in enumerate(k.split('-')[1:]):
        en = int(en)
        new_vq_vectors[e].append(args.model.ord_vq_layers[e].embedding.weight[en].unsqueeze(0).unsqueeze(0))
    
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)

    for i in range(2):
      #vq_vectors.append(torch.sum(torch.stack(new_vq_vectors[i]),dim=0))
      weights = [v/sum(reinflect_tag.values()) for v in reinflect_tag.values()]
      weights = (torch.tensor(weights).unsqueeze(1).unsqueeze(1).unsqueeze(1)).repeat(1,1,1,150)
      weighted =  (torch.stack(new_vq_vectors[i]) * weights)
      vq_vectors.append(torch.sum(weighted,dim=0))
   
    '''vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[reinflect_tag[0]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >= 2:
      vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[reinflect_tag[1]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >= 4:
      vq_vectors.append(args.model.ord_vq_layers[2].embedding.weight[reinflect_tag[2]].unsqueeze(0).unsqueeze(0))
      vq_vectors.append(args.model.ord_vq_layers[3].embedding.weight[reinflect_tag[3]].unsqueeze(0).unsqueeze(0))
    if args.num_dicts >=8:
      vq_vectors.append(args.model.ord_vq_layers[4].embedding.weight[reinflect_tag[4]].unsqueeze(0).unsqueeze(0))
      vq_vectors.append(args.model.ord_vq_layers[5].embedding.weight[reinflect_tag[5]].unsqueeze(0).unsqueeze(0))
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
        vq_vectors.append(args.model.ord_vq_layers[15].embedding.weight[reinflect_tag[15]].unsqueeze(0).unsqueeze(0))'''


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


def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    #model_id = 'unilstm_4x10_dec100_suffixd512'
    #model_id = 'kl0.1_4x10_dec128_suffixd512'
    model_id = 'bilstm_4x10dec100_suffixd512'
    model_id = 'discretelemma_bilstm_4x10_dec512_suffixd512'
    model_id = 'kl0.1_8x10_dec128_suffixd512'
    model_id = 'kl0.1_16x6_dec128_suffixd512'
    model_id = 'bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'discretelemma_sum_bilstm_8x6_dec512_suffixd64'
    model_id = 'bi_kl0.1_sum_8x6_dec128_suffixd64'
    model_id = 'bi_kl0.1_8x6_dec128_suffixd512'
    model_id = 'discretelemma_bilstm_8x6_dec512_suffixd512'
    model_id = 'discretelemma_sum_bilstm_8x6_dec512_suffixd64'
    model_id = 'discretelemma_sum_bilstm_4x6_dec512_suffixd128'
    model_id = 'FINAL_bi_kl0.2_5x5_dec256_suffixd300'
    model_id = 'FINAL_bi_kl0.1_2x20_dec256_suffixd300'

    args.lemmadict_emb_num  = 10000

    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.4; args.dec_dropout_out = 0.0
    args.ni = 256; 
    args.enc_nh = 300
    args.dec_nh = 256  
    args.embedding_dim = args.enc_nh
    args.beta = 0.5
    args.nz = 128; 
    args.num_dicts = 2; 
    args.outcat=0; 
    args.orddict_emb_num =  20
    args.incat = args.enc_nh; 

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

    if 'bi_kl' in model_id and 'sum' in model_id:
        from vqvae_kl_bi_sum import VQVAE
        args.incat = int(args.enc_nh/args.num_dicts)        
        args.model_type = 'bi_kl_sum'


    model_path, model_vocab  = get_model_info(model_id)
    args.model_path = model_path
    args.model_id = model_id
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
    args.beta = 0.5
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args



args = config()
model_id = args.model_id
gold_tags_dict = dict(); found_tags_dict = dict(); found_tags_dict_r = dict()
with open('data/sigmorphon2016/turkish-task2-train', 'r') as reader:
  for line in reader:
    split_line = line.strip().split('\t')
    tag  = split_line[0]
    word = split_line[1]
    gold_tags_dict[word] = tag
    tag  = split_line[2]
    word = split_line[3]
    gold_tags_dict[word] = tag

with open('data/sigmorphon2016/turkish-task2-test', 'r') as reader:
  for line in reader:
    split_line = line.strip().split('\t')
    tag  = split_line[0]
    word = split_line[1]
    gold_tags_dict[word] = tag
    tag  = split_line[2]
    word = split_line[3]
    gold_tags_dict[word] = tag


with open('data/sigmorphon2016/turkish-task2-dev', 'r') as reader:
  for line in reader:
    split_line = line.strip().split('\t')
    tag  = split_line[0]
    word = split_line[1]
    gold_tags_dict[word] = tag
    tag  = split_line[2]
    word = split_line[3]
    gold_tags_dict[word] = tag


c=args.num_dicts
id_tags = dict()
with open('/home/mugekural/dev/git/trmor/model/vqvae/results/training/55204_instances/2noloaded_FINAL_bi_kl0.2_2x20_dec256_suffixd300/suffix_codes.json', 'r') as f:
  modeldata = json.load(f)
tag_counts=defaultdict(lambda: 0)
for key,values in modeldata.items():
  for value in values:
    instance =value[3:-4]
    tag = gold_tags_dict[instance]
    tag_counts[tag] += 1
    if key not in id_tags:
      id_tags[key]=defaultdict(lambda:0)
    id_tags[key][tag] +=1
  for value in values:  
    instance =value[3:-4]
    tag = gold_tags_dict[instance]
    if tag not in found_tags_dict_r:
      found_tags_dict_r[tag] = dict()
    if key not in found_tags_dict_r[tag]:
      found_tags_dict_r[tag][key] = 0
    found_tags_dict_r[tag][key] +=1
for k,v in found_tags_dict_r.items():
    x = found_tags_dict_r[k]
    found_tags_dict_r[k] = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

#with open(args.logdir+model_id+'_id_to_tags_'+str(c)+'.json', 'w') as f:
#    f.write(json.dumps(id_tags))

#with open(args.logdir+model_id+'_tags_to_id_'+str(c)+'.json', 'w') as f:
#    f.write(json.dumps(found_tags_dict_r))

with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
  todo = []
  found_tags = 0; not_found_tags = 0
  for line in reader:
    split_line = line.split('\t')
    asked_tag  = split_line[1]
    inflected_word = split_line[0]
    reinflected_word = split_line[2].strip()
    if asked_tag not in found_tags_dict_r:
      not_found_tags+=1
    else:
      found_tags+=1
      todo.append((inflected_word +' ||| '+ json.dumps(found_tags_dict_r[asked_tag])+' ||| '+ asked_tag +' ||| '+ reinflected_word))
print(found_tags,'  found.', not_found_tags,' not found.')
with open(args.logdir+model_id+'_turkish-task3-test_todo_'+str(c), 'w') as writer:
  for i in todo:
    writer.write(str(i)+'\n')



true = 0; false = 0 
with torch.no_grad():
    with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected', 'w') as writer:
        with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_true', 'w') as writer_true:
            with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_false', 'w') as writer_false:
                with open(args.logdir+args.model_id+'_turkish-task3-test_todo_'+str(args.num_dicts), 'r') as reader:
                    for i,line in enumerate(reader):
                        inflected_word, _tag, tag_name, gold_reinflection = line.strip().split('|||')
                        inflected_word = inflected_word.strip()
                        tag_name = tag_name.strip()
                        tags = json.loads(_tag.strip())
                        gold_reinflection = gold_reinflection.strip()
                        key = tags#[0]
                        #key = [int(s) for s in key.split('-')[1:]]
                        #if args.model_type =='discrete':
                        #  key = key[1:]
                        reinflected_word, vq_code =  reinflect(args, inflected_word,key)
                        reinflected_word = reinflected_word[:-4]
                        writer.write(inflected_word +'\t'+ tag_name+'\t'+reinflected_word + '\n')
                        if reinflected_word == gold_reinflection:
                          true+=1
                          writer_true.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\n')
                        else:
                          false+=1
                          writer_false.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ str(vq_code)+'\n')
                print('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/(true+false)))
                writer_true.write('true %d, false %d, total %d, accuracy: %.2f' % (true,false, true+false, true/(true+false)))




