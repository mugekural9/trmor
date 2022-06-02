# -----------------------------------------------------------
# Date:        2022/02/15
# Author:      Muge Kural
# Description: Word copier for trained VQVAE model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from vqvae import VQVAE
from common.utils import *
import argparse, torch, json,  os

def reinflect(args, inflected_word, reinflect_tag):
    x = torch.tensor([args.vocab.word2id['<s>']] + args.vocab.encode_sentence(inflected_word) + [args.vocab.word2id['</s>']]).unsqueeze(0)
    _, _, _, fwd_fhs, bck_fhs = args.model.encoder(x)
    root_fhs =  args.model.linear_root(fwd_fhs)
    root_z = torch.sigmoid(root_fhs)
    
    bosid = args.vocab.word2id['<s>']
    input = torch.tensor(bosid)
    sft = nn.Softmax(dim=1)
    j1,j2 = reinflect_tag
    vq_vectors = []
    vq_vectors.append(args.model.ord_vq_layers[0].embedding.weight[j1].unsqueeze(0).unsqueeze(0))
    vq_vectors.append(args.model.ord_vq_layers[1].embedding.weight[j2].unsqueeze(0).unsqueeze(0))
    suffix_z = torch.cat(vq_vectors,dim=2)

    batch_size, seq_len, _ = root_z.size()
    z_ = suffix_z.expand(batch_size, seq_len, args.model.decoder.incat)
    root_z = root_z.permute((1,0,2))

    c_init = root_z 
    h_init = torch.tanh(c_init)
    decoder_hidden = (c_init, h_init)
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
            return(''.join(copied), (j1,j2))
            break
    return(''.join(copied), (j1,j2))


#ins = torch.cat(vq_vectors[1:],dim=1)
#output, (last_state, last_cell) = args.model.suffix_lstm(ins)
#last_state = last_state.squeeze(0).unsqueeze(1)
#for i in range(2, len(vq_vectors)):
#    vq_vectors[1] += vq_vectors[i]

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = '2x30_suffixd512'
    model_path, model_vocab  = get_model_info(model_id)
    args.model_id = model_id#[5:]
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
    args.dec_nh = 128 #args.enc_nh; 
    args.embedding_dim = args.enc_nh; 
    args.beta = 0.5
    args.rootdict_emb_num = 3000
    args.orddict_emb_num  = 30
    args.rootdict_emb_dim = 512;  args.nz = 512; 
    args.num_dicts = 2; args.outcat=0; args.incat = 512
    args.model = VQVAE(args, args.vocab, model_init, emb_init, dict_assemble_type='sum_and_concat')
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args


def main():
    args = config()
    with torch.no_grad():
        with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected', 'w') as writer:
            with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_true', 'w') as writer_true:
                with open(args.logdir+args.model_id+'_turkish-task3-test_reinflected_false', 'w') as writer_false:
                    jss =[]
                    '''for k in range(8):
                        js.append([])
                    for k in range(8):
                        with open(args.logdir+args.model_id+'_turkish-task3-test_todo_LSTM_'+str(k), 'r') as reader:
                            for line in reader:
                                inflected_word, _tag, tag_name, gold_reinflection = line.strip().split('|||')
                                inflected_word = inflected_word.strip()
                                tag_name = tag_name.strip()
                                tags = json.loads(_tag.strip())
                                gold_reinflection = gold_reinflection.strip()
                                for key,val in tags.items():
                                    js[k].append(int(key))
                                    break'''

                    with open(args.logdir+args.model_id+'_turkish-task3-test_todo_LSTM_'+str(2), 'r') as reader:
                        for line in reader:
                            inflected_word, _tag, tag_name, gold_reinflection = line.strip().split('|||')
                            inflected_word = inflected_word.strip()
                            tag_name = tag_name.strip()
                            tags = json.loads(_tag.strip())
                            gold_reinflection = gold_reinflection.strip()
                            for key,val in tags.items():
                                jss.append([int(s) for s in key.split('-')[1:]])
                                break
                        
            
                    with open(args.logdir+args.model_id+'_turkish-task3-test_todo_LSTM_2', 'r') as reader:
                        for i,line in enumerate(reader):
                            inflected_word, _tag, tag_name, gold_reinflection = line.strip().split('|||')
                            inflected_word = inflected_word.strip()
                            tag_name = tag_name.strip()
                            tags = json.loads(_tag.strip())
                            gold_reinflection = gold_reinflection.strip()


                            for key,val in tags.items():
                                #key = jss[i]
                                key = [int(s) for s in key.split('-')[1:]]
                                reinflected_word, vq_code =  reinflect(args, inflected_word, key)#[k[i] for k in js])
                                reinflected_word = reinflected_word[:-4]
                                #writer.write(inflected_word +'\t'+tag_name+'\t'+reinflected_word + '\t'+ _tag +'\n')
                                #writer.write(inflected_word +'\t'+str(vq_code)+'\t'+tag_name+'\t'+reinflected_word +'\n')
                                writer.write(inflected_word +'\t'+ tag_name+'\t'+reinflected_word + '\n')
                                print(reinflected_word)
                                if reinflected_word == gold_reinflection:
                                    writer_true.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\n')
                                else:
                                    writer_false.write(inflected_word +'\t'+gold_reinflection + '\t'+reinflected_word+'\t'+ str(vq_code)+'\n')
                                #break
if __name__=="__main__":
    main()



