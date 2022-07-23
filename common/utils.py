"""
Logger class files
"""
import sys
import torch
import random
import torch.nn as nn
from numpy import dot, zeros
from numpy.linalg import matrix_rank, norm


def get_model_info(id, lang=None):
    if id == 'ae_001':
        model_path  = 'model/vqvae/results/training/10000_instances/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/10000_instances/surf_vocab.json'   
    if id == 'ae_002':
        model_path  = 'model/vqvae/results/training/50000_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/surf_vocab.json'  
    
    if id == 'ae_finnish':
        model_path  = 'model/vqvae/results/training/finnish/55000_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/finnish/55000_instances/surf_vocab.json'  

    if id == 'ae_turkish':
        model_path  = 'model/vqvae/results/training/turkish/55204_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/55204_instances/surf_vocab.json'  

    if id == 'ae_turkish_unsup_660':
        model_path  = 'model/vqvae/results/training/turkish/unsup/55000_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/unsup/55000_instances/surf_vocab.json'  
   

    if id == 'ae_turkish_unsup_660_17epc':
        model_path  = 'model/vqvae/results/training/turkish/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/unsup/55000_instances/surf_vocab.json'  

    if id == 'ae_finnish_unsup_660':
        model_path  = 'model/vqvae/results/training/finnish/unsup/55000_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/finnish/unsup/55000_instances/surf_vocab.json'  
  
    if id == 'ae_hungarian_unsup_660':
        model_path  = 'model/vqvae/results/training/hungarian/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/hungarian/unsup/55000_instances/surf_vocab.json'  
  
  
    if id == 'ae_maltese_unsup_660':
        model_path  = 'model/vqvae/results/training/maltese/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/maltese/unsup/55000_instances/surf_vocab.json'  
   
    if id == 'ae_navajo_unsup_240':
        model_path  = 'model/vqvae/results/training/navajo/unsup/32109_instances/7epochs.pt'
        model_vocab = 'model/vqvae/results/training/navajo/unsup/32109_instances/surf_vocab.json'  
   

    if id == 'ae_navajo_unsup_660':
        model_path  = 'model/vqvae/results/training/navajo/unsup/32109_instances/6epochs.pt'
        model_vocab = 'model/vqvae/results/training/navajo/unsup/32109_instances/surf_vocab.json'  
   

    if id == 'ae_russian_unsup_660':
        model_path  = 'model/vqvae/results/training/russian/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/russian/unsup/55000_instances/surf_vocab.json'  
   
    if id == 'ae_arabic_unsup_660':
        model_path  = 'model/vqvae/results/training/arabic/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/arabic/unsup/55000_instances/surf_vocab.json'  
   
    if id == 'ae_german_unsup_660':
        model_path  = 'model/vqvae/results/training/german/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/german/unsup/55000_instances/surf_vocab.json'  
  

    #### SIGMORPHON2018
    if id == 'sigmorphon2018_ae_turkish_unsup_660':
        model_path  = 'model/vqvae/results/training/turkish/sig2018/10000_instances/50epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/sig2018/10000_instances/surf_vocab.json'  

    if id == 'sigmorphon2018_ae_turkish_unsup_1320':
        model_path  = 'model/vqvae/results/training/turkish/sig2018/10000_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/sig2018/10000_instances/surf_vocab.json' 

    if id == 'sigmorphon2018_ae_turkish_unsup_360':
        model_path  = 'model/vqvae/results/training/turkish/sig2018/10000_instances/51epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/sig2018/10000_instances/surf_vocab.json' 



    if id == 'sigmorphon2018_ae_spanish_unsup_660':
        model_path  = 'model/vqvae/results/training/spanish/sig2018/10000_instances/50epochs.pt'
        model_vocab = 'model/vqvae/results/training/spanish/sig2018/10000_instances/surf_vocab.json'  

    if id == 'sigmorphon2018_ae_finnish_unsup_660':
        model_path  = 'model/vqvae/results/training/finnish/sig2018/10000_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/finnish/sig2018/10000_instances/surf_vocab.json'  









    ## LATE SUPERVISION

    if lang == 'finnish':    
        if id == 'batchsize128_beta_0.2_5x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_190'
    
        if id == 'batchsize128_beta_0.2_6x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_6x6_dec256_suffixd300/200epochs.pt_190'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_6x6_dec256_suffixd300/surf_vocab.json'  
    

    if lang == 'turkish':    
        if id == 'beta_0.1_5x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/55204_instances/beta0.1_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_190'
            model_vocab = 'model/vqvae/results/training/55204_instances/beta0.1_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
   
        if id == 'beta_0.5_5x6_bi_kl_0.2_epc160':
            model_path  = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.5_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_160'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.5_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
     
        if id == 'beta_0.2_5x6_bi_kl_0.2_epc170':
            model_path  = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.2_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_170'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.2_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
     
        ## Semisup vqvae
        if id == 'semisup_batchsize128_beta_0.2_11x6_bi_kl_0.2_epc180':
            model_path = 'model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/200epochs.pt_180'
            model_vocab = 'model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append('model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/'+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    
        if id == '12k_semisup_batchsize128_beta_0.2_11x6_bi_kl_0.2_epc180':
            model_path = 'model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/200epochs.pt_180'
            model_vocab = 'model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append('model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/'+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs


        if id == 'late-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc140':
            model_pfx = 'model/vqvae/results/training/turkish/late-supervision/55204_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '300epochs.pt_140'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    

        if id == 'late-supervision_batchsize128_beta_0.1_11x6_bi_kl_0.1_epc120':
            model_pfx = 'model/vqvae/results/training/turkish/late-supervision/55204_instances/batchsize128_beta0.1_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '300epochs.pt_120'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs

        if id == 'early-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc200':
            model_pfx = 'model/vqvae/results/training/turkish/early-supervision/12798_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '301epochs.pt_15'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs

        if id == 'early-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc200':
            model_pfx = 'model/vqvae/results/training/turkish/early-supervision/12798_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '301epochs.pt_10'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    
        if id == 'early-supervision_batchsize128_beta_0.2_11x90_bi_kl_0.1_epc301':
            model_pfx = 'model/vqvae/results/training/turkish/early-supervision/12798_instances/batchsize128_beta0.2_bi_kl0.1_11x90_dec256_suffixd660/'
            model_path  = model_pfx + '301epochs.pt_240'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs




  
    return model_path, model_vocab

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "w")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)


def plot_curves(task, bmodel, fig, ax, trn_loss_values, val_loss_values, style, ylabel):
    ax.plot(range(len(trn_loss_values)), trn_loss_values, style, label=bmodel+'_trn')
    ax.plot(range(len(val_loss_values)), val_loss_values, style,label=bmodel+'_val')
    if ylabel != 'acc': # hack for clean picture
        leg = ax.legend() #(loc='upper right', bbox_to_anchor=(0.5, 1.35), ncol=3)
        ax.set_title(task,loc='left')
    if ylabel != 'loss':
        ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)       


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_li_vectors(dim, R):
    r = matrix_rank(R) 
    index = zeros( r ) #this will save the positions of the li columns in the matrix
    counter = 0
    index[0] = 0 #without loss of generality we pick the first column as linearly independent
    j = 0 #therefore the second index is simply 0

    for i in range(R.shape[1]): #loop over the columns
        if i != j: #if the two columns are not the same
            inner_product = dot( R[:,i], R[:,j] ) #compute the scalar product
            norm_i = norm(R[:,i]) #compute norms
            norm_j = norm(R[:,j])

            #inner product and the product of the norms are equal only if the two vectors are parallel
            #therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
            if abs(inner_product - norm_j * norm_i) > 1e-4:
                counter += 1 #counter is incremented
                index[counter] = i #index is saved
                j = i #j is refreshed
            #do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!
    R_independent = zeros((r, dim))
    i = 0
    #now save everything in a new matrix
    while( i < dim ):
        #R_independent[i,:] = R[index[i],:] 
        R_independent[:,i] = R[:,index[i]] 
        i += 1
    return R_independent


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


'''lines = []
with open('trn_4x10.txt', 'r') as reader:
    for line in reader:
        lines.append(line)

random.shuffle(lines)

with open('trn_4x10_shuffled.txt', 'w') as writer:
    for line in lines[:10000]:
        writer.write(line)

with open('val_4x10_shuffled.txt', 'w') as writer:
    for line in lines[10000:]:
        writer.write(line)'''