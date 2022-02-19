"""
Logger class files
"""
import sys
import torch
import torch.nn as nn

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

from numpy import dot, zeros
from numpy.linalg import matrix_rank, norm

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

def get_model_info(id):
    # ae
    if id == 'ae_1':
        model_path  = 'model/ae/results/training/50000_instances/15epochs.pt'
        model_vocab = 'model/ae/results/training/50000_instances/surf_vocab.json'
    elif id == 'ae_2':
        model_path  = 'model/ae/results/training/582000_instances/5epochs.pt'
        model_vocab = 'model/ae/results/training/582000_instances/surf_vocab.json'
    elif id == 'ae_3':
        model_path  = 'model/ae/results/training/617298_instances/5epochs.pt'
        model_vocab = 'model/ae/results/training/617298_instances/surf_vocab.json'
    # vae
    if id == 'vae_1':
        model_path  = 'model/vae/results/training/582000_instances/10epochs.pt'
        model_vocab = 'model/vae/results/training/582000_instances/surf_vocab.json'
    elif id == 'vae_2': 
        model_path = 'trmor_agg1_kls0.10_warm10_2612_0.pt'
        model_vocab = 'model/vae/results/training/582000_instances/surf_vocab.json'
    elif id == 'vae_5': 
        model_path  = 'model/vae/results/training/617298_instances/10epochs.pt'
        model_vocab = 'model/vae/results/training/617298_instances/surf_vocab.json'
    elif id == 'vae_6': 
        model_path  = 'model/vae/results/training/617298_instances/12epochs.pt'
        model_vocab = 'model/vae/results/training/617298_instances/surf_vocab.json'
    elif id == 'vae_7': 
        model_path  = 'model/vae/results/training/50000_instances/10epochs.pt'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'
    elif id == 'vae_8': 
        model_path  = 'model/vae/results/training/20000_instances/30epochs.pt'
        model_vocab = 'model/vae/results/training/20000_instances/surf_vocab.json'
    elif id == 'vae_9': 
        model_path  = 'model/vae/results/training/20000_instances/20epochs.pt'
        model_vocab = 'model/vae/results/training/20000_instances/surf_vocab.json'
    elif id == 'vae_10': 
        model_path  = 'model/vae/results/training/50000_instances/15epochs.pt'
        model_vocab = 'model/ae/results/training/50000_instances/surf_vocab.json'
    #charlm
    elif id == 'charlm_1': 
        model_path  = 'model/charlm/results/training/582000_instances/35epochs.pt'
        model_vocab = 'model/charlm/results/training/582000_instances/surf_vocab.json'
    elif id == 'charlm_2': 
        model_path  = 'model/charlm/results/training/617298_instances/35epochs.pt'
        model_vocab = 'model/charlm/results/training/617298_instances/surf_vocab.json'
    elif id == 'charlm_3': 
        model_path  = 'model/charlm/results/training/617298_instances/30epochs.pt'
        model_vocab = 'model/charlm/results/training/617298_instances/surf_vocab.json'
    #vqvae
    elif id == 'vqvae_1': 
        model_path  = 'model/vqvae/results/training/50000_instances/100epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/surf_vocab.json'
    return model_path, model_vocab

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
