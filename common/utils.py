"""
Logger class files
"""
import sys
import torch
import random
import torch.nn as nn
from numpy import dot, zeros
from numpy.linalg import matrix_rank, norm


def get_model_info(id):
    if id == 'ae_001':
        model_path  = 'model/vqvae/results/training/10000_instances/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/10000_instances/surf_vocab.json'   
    if id == 'ae_002':
        model_path  = 'model/vqvae/results/training/50000_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/surf_vocab.json'  
    
    if id == 'ae_unilstm':
        model_path  = 'model/vqvae/results/training/12229_instances/25epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/surf_vocab.json'  
    if id == 'ae_bilstm':
        model_path  = 'model/vqvae/results/training/12229_instances/22epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/surf_vocab.json'  
    if id == 'ae_bilstm_zhou':
        model_path  = 'model/vqvae/results/training/25031_instances/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/surf_vocab.json'  
    if id == 'ae_unilstm_zhou':
        model_path  = 'model/vqvae/results/training/25031_instances/21epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/surf_vocab.json'  

    if id == 'ae_bilstm_29k_zhou':
        model_path  = 'model/vqvae/results/training/28794_instances/22epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/surf_vocab.json'  
    if id == 'ae_unilstm_29k_zhou':
        model_path  = 'model/vqvae/results/training/28794_instances/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/surf_vocab.json'  



    if id == 'unilstm_4x10_dec100_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/unilstm_4x10_dec100_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/unilstm_4x10_dec100_suffixd512/surf_vocab.json'  
    
    if id == 'kl0.1_4x10_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/kl0.1_4x10_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/kl0.1_4x10_dec128_suffixd512/surf_vocab.json'  
    
    #if id == 'kl0.1_8x10_dec128_suffixd512':
    #    model_path  = 'model/vqvae/results/training/12229_instances/kl0.1_8x10_dec128_suffixd512/200epochs.pt'
    #    model_vocab = 'model/vqvae/results/training/12229_instances/kl0.1_8x10_dec128_suffixd512/surf_vocab.json'  


    if id == 'bilstm_4x10dec100_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/bilstm_4x10dec100_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/bilstm_4x10dec100_suffixd512/surf_vocab.json'  
    
    if id == 'discretelemma_bilstm_4x10_dec512_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/discretelemma_bilstm_4x10_dec512_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/discretelemma_bilstm_4x10_dec512_suffixd512/surf_vocab.json'  
    
    if id == 'kl0.2_4x10_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/kl0.2_4x10_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/kl0.2_4x10_dec128_suffixd512/surf_vocab.json'  


    if id == 'kl0.1_8x10_dec150_suffixd512':
        model_path  = 'model/vqvae/results/training/12229_instances/kl0.1_8x10_dec150_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/12229_instances/kl0.1_8x10_dec150_suffixd512/surf_vocab.json'  


    if id == 'discretelemma_bilstm_4x10_dec512_suffixd512':
        model_path  = 'model/vqvae/results/training/25031_instances/discretelemma_bilstm_4x10_dec512_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/discretelemma_bilstm_4x10_dec512_suffixd512/surf_vocab.json'  


    if id == 'kl0.1_8x10_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/25031_instances/kl0.1_8x10_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/kl0.1_8x10_dec128_suffixd512/surf_vocab.json'  

    if id == 'kl0.1_8x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/25031_instances/kl0.1_8x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/kl0.1_8x6_dec128_suffixd512/surf_vocab.json'  
    
    if id == 'kl0.1_16x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/25031_instances/kl0.1_16x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/25031_instances/kl0.1_16x6_dec128_suffixd512/surf_vocab.json'  

    if id == 'kl0.1_8x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/kl0.1_8x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/kl0.1_8x6_dec128_suffixd512/surf_vocab.json'  

    if id == 'kl0.1_8x10_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/kl0.1_8x10_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/kl0.1_8x10_dec128_suffixd512/surf_vocab.json'  

    if id == 'bi_kl0.1_8x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/bi_kl0.1_8x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/bi_kl0.1_8x6_dec128_suffixd512/surf_vocab.json'  

    if id == 'discretelemma_bilstm_8x6_dec512_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/discretelemma_bilstm_8x6_dec512_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/discretelemma_bilstm_8x6_dec512_suffixd512/surf_vocab.json'  

    if id == 'bi_kl0.1_16x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/bi_kl0.1_16x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/bi_kl0.1_16x6_dec128_suffixd512/surf_vocab.json'  

    if id == 'bi_kl0.2_8x6_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/bi_kl0.2_8x6_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/bi_kl0.2_8x6_dec128_suffixd512/surf_vocab.json'  

    if id == 'bi_kl0.1_8x4_dec128_suffixd512':
        model_path  = 'model/vqvae/results/training/28794_instances/bi_kl0.1_8x4_dec128_suffixd512/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/28794_instances/bi_kl0.1_8x4_dec128_suffixd512/surf_vocab.json'  


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