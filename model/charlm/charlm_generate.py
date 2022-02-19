# -----------------------------------------------------------
# Date:        2021/12/17 
# Author:      Muge Kural
# Description: Random word generator for trained charLM model
# -----------------------------------------------------------

from common.vocab import VocabEntry
from charlm import CharLM
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os


def generate(args):
    def init_hidden(model, bsz):
        weight = next(model.parameters())
        return (weight.new_zeros(1, bsz, model.nh),
                weight.new_zeros(1, bsz, model.nh))

    bosid = args.vocab.word2id['<s>'] 
    input = torch.tensor([bosid]).unsqueeze(0)
    word = []
    sft = nn.Softmax(dim=1)
    i = 0; max_length = 20
    decoder_hidden = init_hidden(args.model,1)
    while i < max_length:
        i +=1
        word_embed = args.model.embed(input)
        output, decoder_hidden = args.model.lstm(word_embed, decoder_hidden)
        output_logits = args.model.pred_linear(output).squeeze(1)
        input = torch.multinomial(sft(output_logits), num_samples=1) # sample
        char = args.vocab.id2word(input.item())
        word.append(char)
        if char == '</s>':
            word = ''.join(word)
            print(word)
            return word

def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'charlm_3'
    model_path, model_vocab  = get_model_info(model_id)
    # logging
    args.logdir = 'model/charlm/results/generation/'+model_id+'/'
    args.logfile = args.logdir + '/samples.txt'
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

    args.ni = 512; args.nh = 1024
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.model = CharLM(args, args.vocab, model_init, emb_init)

    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.eval()
    return args

def main():
    args = config()
    # generate random words
    with open(args.logfile, "w") as f:
        for i in range(100):
            word = generate(args)
            f.write(word + "\n")

if __name__=="__main__":
    main()



