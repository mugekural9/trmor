"""
Logger class files
"""
import sys
import torch
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
    if id == 'ae_004':
        model_path  = 'model/vqvae/results/training/12798_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/12798_instances/surf_vocab.json'  
    
    if id == 'vqvae_1x10000_3x8':
        model_path  = 'model/vqvae/results/training/50000_instances/1x10000_3x8/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/1x10000_3x8/surf_vocab.json'  
    if id == 'vqvae_1x10000_4x6':
        model_path  = 'model/vqvae/results/training/50000_instances/1x10000_4x6/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/1x10000_4x6/surf_vocab.json'  
    if id == 'vqvae_1x10000_8x6':
        model_path  = 'model/vqvae/results/training/50000_instances/1x10000_8x6/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/1x10000_8x6/surf_vocab.json'  
   
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
