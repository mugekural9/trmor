from modules import VAE, LSTMEncoder, LSTMDecoder
from data import MonoTextData, read_trndata_makevocab, read_valdata, get_batches
import torch
import torch.nn as nn
import argparse
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd  
from numpy import save
from sklearn.datasets import fetch_openml

def make_data(trnsize, batchsize):
    # Read data and get batches...
    surface_vocab = MonoTextData('trmor_data/trmor2018.trn', label=False).vocab
    trndata, feature_vocab, data_str = read_trndata_makevocab('trmor_data/trmor2018.trn', trnsize, surface_vocab) # data len:50000
    vlddata = read_valdata('trmor_data/trmor2018.val', feature_vocab, surface_vocab)     # 5000
    tstdata = read_valdata('trmor_data/trmor2018.tst', feature_vocab, surface_vocab)     # 5981
    trn_batches, _ = get_batches(trndata, surface_vocab, feature_vocab, batchsize) 
    vld_batches, _ = get_batches(vlddata, surface_vocab, feature_vocab, batchsize) 
    tst_batches, _ = get_batches(tstdata, surface_vocab, feature_vocab, batchsize) 

    return (trn_batches, vld_batches, tst_batches), surface_vocab, feature_vocab, data_str

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
args.ni = 512
args.enc_nh = 1024
args.dec_nh = 1024
args.nz = 32
args.enc_dropout_in = 0.0
args.dec_dropout_in = 0.0
args.dec_dropout_out = 0.0
args.trnsize = 50000
args.batchsize = 1

# DATA
data, surface_vocab, feature_vocab,  (surf_data_str, feat_data_str)  = make_data(args.trnsize, args.batchsize)
trn, val, tst = data

# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  #feature_vocab for scratch preloading
model = VAE(encoder, decoder, args)
model.encoder.mode = 's2s'

# Load model
args.basemodel = "models/vae/vae_trmor_agg1_kls0.10_warm10_0_9.pt"
model.load_state_dict(torch.load(args.basemodel))
print('Model weights loaded from ... ', args.basemodel)
model.to(args.device)

x = []
numpoints = 200
indices = list(range(len(trn)))

for i, idx in enumerate(indices):
    surf, feat= trn[idx]
    # (batchsize, tx)
    surf = surf.t().to(args.device)
    # (batchsize, ty)
    feat = feat.t().to(args.device)
    # z: (batchsize, 1, nz)
    z, KL = model.encode(surf, 1)
    # (batchsize, nz)
    x.append(z.squeeze(1).cpu().detach())

# (ninstances, hiddendim)
X = np.array(torch.cat((x))) # 50000 data
save('vae_encoded_data.npy', X)
#save('surf_data_str.npy', surf_data_str)

df = pd.DataFrame()
df['y'] = np.array(surf_data_str)[:numpoints]
# (batchsize, n_components:reduced_dim)
tsne_results = TSNE(n_components=2, verbose=1).fit_transform(X[:numpoints])
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #hue="y",
    #palette=sns.color_palette("hls", numpoints),
    data=df,
    legend="full",
    alpha=0.3
)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(df['tsne-2d-one'],  df['tsne-2d-two'], df['y'], plt.gca()) 
plt.savefig('vae_tsne.png')

    