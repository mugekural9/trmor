import torch, argparse, matplotlib, random
import pandas as pd  
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import VAE, LSTMEncoder, LSTMDecoder
from data import MonoTextData, read_trndata_makevocab, read_valdata, get_batches, build_data
from numpy import save
from sklearn.datasets import fetch_openml
from utils import uniform_initializer
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
from numpy import dot
from numpy.linalg import norm
matplotlib.use('Agg')

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
args.trnsize = 57769
args.batchsize = 1
args.trndata = 'trmor_data/polarity_trn.txt' # 'trmor_data/trmor2018.trn'
args.valdata = 'trmor_data/polarity_val.txt'
args.tstdata = 'trmor_data/trmor2018.tst'
args.seq_to_no_pad = 'surface'
#t-sne config
numpoints = 52000
args.num_annotate = 0; args.annotate_label = ''

# DATA
_ , data, vocab, args.freqdict, args.freqstagsdict = build_data(args)
trn, val, tst = data
surface_vocab, feature_vocab, pos_vocab, polar_vocab = vocab #fix this rebuilding vocab issue!
# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  
model = VAE(encoder, decoder, args)
model.encoder.is_reparam = False
# surf2pos probing
model.decoder = None
model.encoder.linear = None
model.probe_linear = nn.Linear(args.enc_nh, len(polar_vocab), bias=False)
# Load model
'''
args.bmodel = 'ae'
if args.bmodel =='vae':
    args.basemodel = "models/vae/vae_trmor_agg1_kls0.10_warm10_0_9.pt9"
    figname = 'z_space_vae_tsne.png'
elif args.bmodel =='ae':
    args.basemodel = "models/ae/ae_trmor_agg0_kls0.10_warm10_0_9.pt9"
    figname = 'z_space_ae_tsne.png'
'''
# surf2pos probing
args.basemodel = "models/surf2polar/5926_polarity_instances_from_random_100epochs.pt"
figname = '5926instances_polar_projections_space_surf2polar-probe-random_tsne.png'
model.load_state_dict(torch.load(args.basemodel))
print('Model weights loaded from ... ', args.basemodel)
model.to(args.device)
model.eval()

# projection a vector on b vector
# projbA =  (a.b / ||b||^2). b # see https://en.wikipedia.org/wiki/Vector_projection
def vector_projection(a, b):
    b_norm = np.sqrt(sum(b**2))    
    proj_of_a_on_b = (np.dot(a, b)/b_norm**2)*b
    return proj_of_a_on_b

# cossimilarity =  (a.b / ||a|| ||b||)  # see https://en.wikipedia.org/wiki/Cosine_similarity
def cos_similarity(a, b):
    result = dot(a, b)/(norm(a)*norm(b))
    return result
     

x = []; hx = []; posdata = []; polardata = []
root_strs = []; surf_strs = []; feat_strs = []
predposdata = []; predpolardata = []
indices = list(range(len(trn)))
for i, idx in enumerate(indices):
    # (batchsize, t)
    surf, feat, pos, root, polar = trn[idx]
    # z: (batchsize, 1, nz), last_states: (1, 1, enc_nh)
    z, KL, last_states = model.encode(surf, 1)
    
    # surf2pos probing
    # last_state_vector: (1024)
    last_state_vector = last_states[0,0,:].cpu().detach()

    sft = nn.Softmax(dim=2)
    # (1, batchsize, vocab_size)
    output_logits = model.probe_linear(last_states)
    pred_polar = torch.argmax(sft(output_logits), 2)
    
    # visualize vector projections on predicted weight vectors
    # pos_weight_vector = model.probe_linear.weight[pred_pos.item()].cpu().detach()
    # vp = vector_projection(last_state_vector, pos_weight_vector)
    # x.append(torch.tensor(vp).unsqueeze(0))
   
    # visualize vector projections on 3 weight vectors
    x.append(output_logits.squeeze(0)[:,:3].cpu().detach())

    #posdata.append(pos.item())
    polardata.append(polar.item())

    #predposdata.append(pred_pos.item())
    predpolardata.append(pred_polar.item())
    root_str = ''.join(surface_vocab.decode_sentence(root.t())[1:-1])
    surf_str = ''.join(surface_vocab.decode_sentence(surf.t())[1:-1]) 
    feat_str = ''.join(feature_vocab.decode_sentence(feat.t())[1:-1]) 
    root_strs.append(root_str)
    surf_strs.append(surf_str)
    feat_strs.append(feat_str)
    hx.append((surf_str, last_state_vector))

# clip first numpoints 
X = np.array(torch.cat((x))) 
print('number of points: ', X.shape)

# (ninstances, hiddendim)
save('5926instances_random_polar_projections_data.npy', X)
save('5926instances_random_feat_strs.npy', feat_strs)
save('5926instances_random_surf_strs.npy', surf_strs)
save('5926instances_random_polardata.npy', polardata)
save('5926instances_random_predpolardata.npy', predpolardata)

'''
## visualizations
X = X[:numpoints]
print('clipped to: ', X.shape)
posdata = posdata[:numpoints]
#posdata.append(7)
root_strs = root_strs[:numpoints]
surf_strs = surf_strs[:numpoints]

tsne_results = TSNE(n_components=2, verbose=1).fit_transform(X)
#{""Verb": 1, "Noun": 2, "Adj": 3, "Num": 4, "Adverb": 5, "Det": 6, "Conj": 7, "Postp": 8, "Pron": 9, "Interj": 10, "Ques": 11, "Dup": 12}
colors = ['red','green','blue','purple', 'gray', 'pink', 'black', 'yellow', 'magenta', 'brown', 'cyan', 'lightblue']
#fig,ax = plt.subplots()
#sc = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=posdata, cmap=matplotlib.colors.ListedColormap(colors))


fig = plt.figure()
ax = plt.axes(projection ="3d")
colors = ['red','green','blue','purple', 'gray', 'pink', 'black', 'yellow', 'magenta', 'brown', 'cyan', 'lightblue']
ax.scatter(X[:,0], X[:,1], X[:,2], c=posdata, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)
def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([surf_strs[n] for n in ind["ind"]]))
                           
    annot.set_text(text)
    #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()


# save the visualization
i = 0
if args.annotate_label == 'root':
    for root,(x,y) in zip(root_strs,tsne_results):
        if i > args.num_annotate:
            break
        else:
            plt.annotate(root, (x, y))
            i +=1
elif args.annotate_label == 'surf':
    for surf,(x,y) in zip(surf_strs,tsne_results):
        if i > args.num_annotate:
            break
        else:
            plt.annotate(surf, (x, y))
            i +=1
elif args.annotate_label == 'feat':
    for feat,(x,y) in zip(feat_strs,tsne_results):
        if i > args.num_annotate:
            break
        else:
            plt.annotate(feat, (x, y))
            i +=1


if model.encoder.is_reparam:
    plt.savefig(str(len(X))+'points_reparam_'+figname)
else:
    plt.savefig(str(len(X))+'points_'+figname)
'''