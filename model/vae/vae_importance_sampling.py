# -----------------------------------------------------------
# Date:        2021/12/28
# Author:      Muge Kural
# Description: Word marginal likelihood estimator for trained VAE model 
# -----------------------------------------------------------

from common.vocab import VocabEntry
from model.vae.vae import VAE
from common.utils import *
import sys, argparse, random, torch, json, matplotlib, os
from evaluation.morph_segmentation.data.data import build_data
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from statistics import stdev, mean

# does the experiment nruns times 
# and returns mean, std of logpx of runs
def run(args, data, sample_from, nsamples, nruns=10, fw=None):
    run_logpx_values =[]
    for r in range(nruns):  
        mu, logvar, _ = args.model.encoder(sample_from)
        param = (mu,logvar,_)
        z = args.model.reparameterize(mu, logvar, nsamples)
        logpx = args.model.nll_iw(data, nsamples, z, param)
        logpx = torch.mean(logpx).item()
        '''mu, logvar, _ = args.model.encoder(sample_from)
        cur = args.model.reparameterize(mu,logvar,nsamples)
        logpx = args.model.eval_complete_ll(data, cur, args.recon_type)
        logpx = torch.mean(logpx).item()'''
        run_logpx_values.append(logpx)
        fw.write("run %d: log p(x) = %.2f \n" % (r, logpx))
    return mean(run_logpx_values), stdev(run_logpx_values)


def config():
    # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'vae_1'
    model_path, model_vocab  = get_model_info(model_id)
    # (a) avg: averages ll over word tokens, (b) sum: adds ll over word tokens
    args.recon_type = 'avg'
    # (a) word_given: sample z from full word, (b) subword_given: sample z from subword
    args.sample_type = 'subword_given'
    # logging
    args.logdir = 'model/vae/results/importance_sampling/'+model_id+'/'+args.recon_type+'/nsamples20000-30000/'+args.sample_type+'/'
    args.logfile = args.logdir + 'importance_sampling.txt'
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
    args.ni = 512; args.nz = 32; 
    args.enc_nh = 1024; args.dec_nh = 1024
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0
    args.model = VAE(args, args.vocab, model_init, emb_init)
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.to(args.device)
    args.model.eval()
    # data
    args.tstdata = 'evaluation/morph_segmentation/data/goldstdsample.tur'
    args.maxtstsize = 5
    args.batch_size = 1
    return args

def main():
    args = config()
    _, batches = build_data(args)
    nsamples_list = [21000,25000,30000] #[35000,40000, 45000, 50000] #[8,16,32,64,128,512,1000,5000,7500,10000,12000,15000,20000]; 
    nruns = 10 
    with torch.no_grad():
        # loop through each word 
        for data in batches:
            sample_from = data
            means = []
            word = ''.join(args.vocab.decode_sentence(data[0][1:-1]))
            print(word)
            f = open(args.logfile+'_'+word, "w")
            # loop through each nsamples 
            for nsamples in nsamples_list:
                f.write("\n---\n")
                # run same experiment multiple times
                mean_runs, stdev_runs = run(args, data, sample_from, nsamples, nruns, f)
                means.append(mean_runs)
                f.write("\nnumber of samples: %d, number of runs: %d, mean: %.4f, stddev: %.4f" % (nsamples, nruns, mean_runs, stdev_runs))
            f.close()

        # loop through each word's subwords, enable if necessary
        if True: 
            for data in batches:
                for i in range(len(data[0])-2, 1, -1):
                    eos  = torch.tensor([2]).to(args.device)
                    subdata = torch.cat([data[0][:i], eos])
                    if args.sample_type == 'word_given':
                        sample_from = data
                    elif args.sample_type == 'subword_given':
                        sample_from = subdata.unsqueeze(0)
                    means = []
                    word = ''.join(args.vocab.decode_sentence(subdata[1:-1]))
                    print(word)
                    f = open(args.logfile+'_'+word, "w")
                    for nsamples in nsamples_list:
                        f.write("\n---\n")
                        mean_runs, stdev_runs = run(args, subdata.unsqueeze(0), sample_from, nsamples, nruns, f)
                        means.append(mean_runs)
                        f.write("\nnumber of samples: %d, number of runs: %d, mean: %.4f, stddev: %.4f" % (nsamples, nruns, mean_runs, stdev_runs))
                    f.close()

if __name__=="__main__":
    main()
