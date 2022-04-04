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
    elif id == 'ae_4':
        model_path  = 'model/ae/results/training/3487_instances/150epochs.pt'
        model_vocab = 'model/ae/results/training/3487_instances/surf_vocab.json'

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
    elif id == 'vae_11': 
        model_path  = 'model/vae/results/training/3487_instances/200epochs.pt'
        model_vocab = 'model/vae/results/training/3487_instances/surf_vocab.json'
    elif id == 'vae_12': 
        model_path  = 'model/vae/results/training/3487_instances/100epochs.pt'
        model_vocab = 'model/vae/results/training/3487_instances/surf_vocab.json'
    elif id == 'vae_neu_1': 
        model_path  = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_agg1_kls0.10_warm10_0_0.pt9'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/trmor_agg0_kls0.10_warm10_0_0_surf_vocab.json'
    elif id == 'vae_neu_2': 
        model_path  = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_agg1_kls0.10_warm10_0_0.pt9'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/trmor_agg1_kls0.10_warm10_0_0_surf_vocab.json'

 
 

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
    elif id == 'charlm_4': 
        model_path  = 'model/charlm/results/training/50000_instances/30epochs.pt'
        model_vocab = 'model/charlm/results/training/50000_instances/surf_vocab.json'
    #vqvae
    elif id == 'vqvae_1': 
        model_path  = 'model/vqvae/results/training/50000_instances/100epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/surf_vocab.json'
    elif id == 'vqvae_2': 
        model_path  = 'model/vqvae/results/training/3487_instances/200epochs.pt'
        model_vocab = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'vqvae_3': 
        model_path  = 'model/vqvae/results/training/3500_instances/multivqvae/499epochs.pt'
        model_vocab = 'model/vqvae/results/training/3500_instances/multivqvae/surf_vocab.json'
    elif id == 'vqvae_4': 
        model_path  = 'model/vqvae/results/training/3500_instances/multivqvae/usedprobes/100epochs.pt'
        model_vocab = 'model/vqvae/results/training/3500_instances/multivqvae/usedprobes//surf_vocab.json'
    elif id == 'mvqvae_001': 
        model_path  = 'model/vqvae/results/training/3487_instances/mvq_001.pt'
        model_vocab = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'mvqvae_003': 
        model_path  = 'model/vqvae/results/training/3487_instances/mvq_003.pt'
        model_vocab = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'mvqvae_with_probes': 
        model_path  = 'model/vqvae/results/training/3487_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'mvqvae_101': 
        model_path  = 'model/vqvae/results/training/3487_instances/250epochs.pt'
        model_vocab = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    #probe
    elif id == 'charlm_4_probe_polar': 
        model_path  = 'evaluation/probing/polarity/results/training/charlm_4_probe/4000_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/polarity/results/training/charlm_4_probe/4000_instances/surf_vocab.json'
        polar_vocab = 'evaluation/probing/polarity/results/training/charlm_4_probe/4000_instances/polar_vocab.json'
        model_vocab = (surf_vocab, polar_vocab)
    elif id == 'charlm_4_probe_pos_tagging': 
        model_path  = 'evaluation/probing/pos_tagging/results/training/charlm_4_probe/14000_instances/300epochs.pt'
        surf_vocab  = 'evaluation/probing/pos_tagging/results/training/charlm_4_probe/14000_instances/surf_vocab.json'
        surfpos_vocab = 'evaluation/probing/pos_tagging/results/training/charlm_4_probe/14000_instances/surfpos_vocab.json'
        model_vocab = (surf_vocab, surfpos_vocab)
    elif id == 'charlm_4_probe_tense': 
        model_path  = 'evaluation/probing/tense/results/training/charlm_4_probe/4000_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/tense/results/training/charlm_4_probe/4000_instances/surf_vocab.json'
        tense_vocab = 'evaluation/probing/tense/results/training/charlm_4_probe/4000_instances/tense_vocab.json'
        model_vocab = (surf_vocab, tense_vocab)
    
    elif id == 'ae_1_probe_pos_tagging': 
        model_path  = 'evaluation/probing/pos_tagging/results/training/ae_1_probe/14000_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/pos_tagging/results/training/ae_1_probe/14000_instances/surf_vocab.json'
        surfpos_vocab = 'evaluation/probing/pos_tagging/results/training/ae_1_probe/14000_instances/surfpos_vocab.json'
        model_vocab = (surf_vocab, surfpos_vocab)
    elif id == 'ae_1_probe_tense_2': 
        model_path  = 'evaluation/probing/tense/results/training/ae_1_probe/3500_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/tense/results/training/ae_1_probe/3500_instances/surf_vocab.json'
        tense_vocab = 'evaluation/probing/tense/results/training/ae_1_probe/3500_instances/tense_vocab.json'
        model_vocab = (surf_vocab, tense_vocab)
    elif id == 'ae_1_probe_polar_2': 
        model_path  = 'evaluation/probing/polarity/results/training/ae_1_probe/3500_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/polarity/results/training/ae_1_probe/3500_instances/surf_vocab.json'
        polar_vocab = 'evaluation/probing/polarity/results/training/ae_1_probe/3500_instances/polar_vocab.json'
        model_vocab = (surf_vocab, polar_vocab)
    elif id == 'ae_1_probe_person_2': 
        model_path  = 'evaluation/probing/person/results/training/ae_1_probe/3500_instances/300epochs.pt'
        surf_vocab  = 'evaluation/probing/person/results/training/ae_1_probe/3500_instances/surf_vocab.json'
        person_vocab = 'evaluation/probing/person/results/training/ae_1_probe/3500_instances/person_vocab.json'
        model_vocab = (surf_vocab, person_vocab)
    elif id == 'ae_1_probe_combined': 
        model_path  = 'evaluation/probing/combined/results/training/ae_1_probe/3487_instances/32dim/300epochs.pt'
        surf_vocab  = 'evaluation/probing/combined/results/training/ae_1_probe/3487_instances/32dim/surf_vocab.json'
        combined_vocab = 'evaluation/probing/combined/results/training/ae_1_probe/3487_instances/32dim/combined_vocab.json'
        model_vocab = (surf_vocab, combined_vocab)

    # 
    elif id == 'ae_for_vqvae_001_probe_person': 
        model_path  = 'evaluation/probing/person/results/training/ae_for_vqvae_001_probe/3487_instances/32dim/300epochs.pt'
        surf_vocab  = 'evaluation/probing/person/results/training/ae_for_vqvae_001_probe/3487_instances/32dim/surf_vocab.json'
        person_vocab = 'evaluation/probing/person/results/training/ae_for_vqvae_001_probe/3487_instances/32dim/person_vocab.json'
        model_vocab = (surf_vocab, person_vocab)

    elif id == 'ae_for_vqvae_001_probe_polarity': 
        model_path  = 'evaluation/probing/polarity/results/training/ae_for_vqvae_001_probe/3487_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/polarity/results/training/ae_for_vqvae_001_probe/3487_instances/surf_vocab.json'
        polarity_vocab = 'evaluation/probing/polarity/results/training/ae_for_vqvae_001_probe/3487_instances/polar_vocab.json'
        model_vocab = (surf_vocab, polarity_vocab)

    elif id == 'ae_for_vqvae_001_probe_rootconcept': 
        model_path  = 'evaluation/probing/root_concept/results/training/ae_for_vqvae_001_probe/3487_instances/300epochs.pt'
        surf_vocab  = 'evaluation/probing/root_concept/results/training/ae_for_vqvae_001_probe/3487_instances/surf_vocab.json'
        root_concept_vocab = 'evaluation/probing/root_concept/results/training/ae_for_vqvae_001_probe/3487_instances/root_concept_vocab.json'
        model_vocab = (surf_vocab, root_concept_vocab)

    elif id == 'ae_for_vqvae_001_probe_tense': 
        model_path  = 'evaluation/probing/tense/results/training/ae_for_vqvae_001_probe/3487_instances/200epochs.pt'
        surf_vocab  = 'evaluation/probing/tense/results/training/ae_for_vqvae_001_probe/3487_instances/surf_vocab.json'
        tense_vocab = 'evaluation/probing/tense/results/training/ae_for_vqvae_001_probe/3487_instances/tense_vocab.json'
        model_vocab = (surf_vocab, tense_vocab)


    elif id == 'ae_for_vqvae_100': 
        model_path  = 'model/vqvae/results/training/100_instances/50epochs.pt'
        model_vocab  = 'model/vqvae/results/training/100_instances/surf_vocab.json'
    elif id == 'ae_for_vqvae_001': 
        model_path  = 'model/vqvae/results/training/3487_instances/ae_001.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'ae_for_vqvae_003': 
        model_path  = 'model/vqvae/results/training/20000_instances/ae_003.pt'
        model_vocab  = 'model/vqvae/results/training/20000_instances/surf_vocab.json'
    elif id == 'ae_for_vqvae_004': 
        model_path  = 'model/vqvae/results/training/10000_instances/ae_004.pt'
        model_vocab  = 'model/vqvae/results/training/10000_instances/surf_vocab.json'

    elif id == 'vqvae_probes': 
        model_path  = 'model/vqvae/results/training/3487_instances/15epochs.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_7_1': 
        model_path  = 'model/vqvae/results/training/3487_instances/7_1.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_16_1': 
        model_path  = 'model/vqvae/results/training/3487_instances/16_1.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_16_3': 
        model_path  = 'model/vqvae/results/training/3487_instances/16_3.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
        
    elif id == 'vqvae_7_2': 
        model_path  = 'model/vqvae/results/training/10000_instances/7_2.pt'
        model_vocab  = 'model/vqvae/results/training/10000_instances/surf_vocab.json'

        
    elif id == 'vqvae_7d_2': 
        model_path  = 'model/vqvae/results/training/10000_instances/7_d2.pt'
        model_vocab  = 'model/vqvae/results/training/10000_instances/surf_vocab.json'

    '''elif id == 'vqvae_8_dict': 
        model_path  = 'model/vqvae/results/training/3487_instances/200epochs.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_1plus8_dict': 
        model_path  = 'model/vqvae/results/training/3487_instances/250epochs.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_9_dict': 
        model_path  = 'model/vqvae/results/training/3487_instances/220epochs.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    
    elif id == 'vqvae_8_dict_sum': 
        model_path  = 'model/vqvae/results/training/3487_instances/8dict_sum.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'
    elif id == 'vqvae_8_dict_concat': 
        model_path  = 'model/vqvae/results/training/3487_instances/8dict_concat.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'

    elif id == 'vqvae_7_dict_enlarged_root': 
        model_path  = 'model/vqvae/results/training/3487_instances/7dict_enlarged_root.pt'
        model_vocab  = 'model/vqvae/results/training/3487_instances/surf_vocab.json'''


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
