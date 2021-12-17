import torch, argparse, matplotlib, random
import pandas as pd  
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import VAE, LSTMEncoder, LSTMDecoder
from data import build_data, log_data
from collections import defaultdict

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

args.maxtrnsize = 100000 
args.maxvalsize = 8329 
args.maxtstsize = 8517

args.batchsize = 1
#args.trndata = 'trmor_data/tense/tense.trn.txt' # 'trmor_data/trmor2018.trn'
#args.valdata = 'trmor_data/tense/tense.val.txt'
#args.tstdata = 'trmor_data/tense_fut.txt'
args.seq_to_no_pad = 'surface'

args.trndata = 'trmor_data/sigmorphon/2018task-1/turkish-train-high' 
args.valdata = 'trmor_data/sigmorphon/2018task-1/turkish-dev' 
args.tstdata = 'trmor_data/sigmorphon/2018task-1/turkish-test' 
args.surface_vocab_file = 'trmor_data/sigmorphon/2018task-1/turkish-train-high'#Turkish.bible.txt' 
args.task = 'sigmorphon2021task2/visualization'
args.bmodel = 'ae_original' 



#t-sne config
numpoints = 52000
args.num_annotate = 0; args.annotate_label = ''

# DATA
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
trnbatches, valbatches, tstbatches = batches
surface_vocab, feature_vocab = vocab # fix this rebuilding vocab issue!


# MODEL
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
encoder = LSTMEncoder(args, len(surface_vocab), model_init, emb_init) 
decoder = LSTMDecoder(args, surface_vocab, model_init, emb_init)  
model = VAE(encoder, decoder, args)
model.encoder.is_reparam = False

# surf2- probing
# model.decoder = None
# model.encoder.linear = None
# model.probe_linear = nn.Linear(args.enc_nh, len(tense_vocab), bias=False)
# Load model

if args.bmodel =='vae':
    args.basemodel = "models/vae/trmor_agg0_kls0.10_warm10_0_911.pt"
    figname = 'z_space_vae_tsne.png'
elif args.bmodel =='ae_original':
    args.basemodel = "models/ae/trmor_agg0_kls0.00_warm10_0_911.pt" #"trmor_agg0_kls0.00_warm10_1_1911.pt"
    figname = 'z_space_ae_tsne.png'

# surf2- probing
# args.basemodel = "models/surf2tense/3814_4tenses_instances_from_ae_300epochs.pt"

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
    result = dot(a, b.t())/(norm(a)*norm(b))
    return result
'''     
hindilerimiz = 'hindilerimiz'
surf_hindilerimiz = [1] + [surface_vocab[char] for char in hindilerimiz] + [2]
surf_hindilerimiz = torch.tensor(surf_hindilerimiz, device='cuda').unsqueeze(0)

hindiler = 'hindiler'
surf_hindiler = [1] + [surface_vocab[char] for char in hindiler] + [2]
surf_hindiler = torch.tensor(surf_hindiler, device='cuda').unsqueeze(0)
z_hindilerimiz, KL, last_states_hindilerimiz, hidden_states_hindilerimiz = model.encode(surf_hindilerimiz, 1)
z_hindiler, KL, last_states_hindiler, hidden_states_hindiler = model.encode(surf_hindiler, 1)'''


x = []; hx = []; zx = []; posdata = []; polardata = []; tensedata = []
root_strs = []; surf_strs = []; feat_strs = []
predposdata = []; predpolardata = []; predtensedata = []
indices = list(range(len(tstbatches)))
random.seed(0); random.shuffle(indices)
#indices = list(range(len(tst)))
for i, idx in enumerate(indices):
    # (batchsize, t)
    #surf, feat, pos, root, polar, tense = trn[idx]
    surf, feat = tstbatches[idx] 
    # z: (batchsize, 1, nz), last_states: (1, 1, enc_nh), last_states: (1, t, enc_nh)
    z, KL, last_states, hidden_states = model.encode(surf, 1)
    z_vector = z[0,0,:].cpu().detach()
    zx.append(z_vector.unsqueeze(0).cpu().detach())
    surf_str = ''.join(surface_vocab.decode_sentence(surf.t())[1:-1]) 
    surf_strs.append(surf_str)
    '''surf_str = surface_vocab.decode_sentence(surf.t()) 
    hx = [];
    _,tx, _ = hidden_states.shape
    surf_strr = ''.join(surf_str)[3:-3]
    forms = dict()
    similarities = dict()
    form_similarities = defaultdict(lambda: 0)
    subsimilarities = defaultdict(lambda: 0)
    print('\n'+surf_strr[:-1])
    fhs = hidden_states[:,tx-1,:].cpu().detach()
    for t in range(1,tx-1):
        #forms[surf_strr[:-t]] = hidden_states[:,tx-t,:].cpu().detach()
        subhs = hidden_states[:,tx-t,:].cpu().detach()
        substr = surf_strr[:-t]
        surf_strs.append(substr)
        cs = cos_similarity(fhs,subhs)
        #print('%s similarity: %.3f' % (substr, cs))
        subsimilarities[substr] = cs
    print('--------')
    for k,v in sorted(subsimilarities.items(), key=lambda item: item[1], reverse=True):
        print('%s : %.3f' % (k,v))
    for form, hs in forms.items():
        for f, h in forms.items():
            cs = cos_similarity(hs,h)
            form_similarities[form] += cs
            if (f,form) not in similarities.keys():
                similarities[(form,f)] = cs
    for k,v in sorted(similarities.items(), key=lambda item: item[1], reverse=True):
        print('%s similarity: %.3f' % (k,v))'''
    #for k,v in sorted(form_similarities.items(), key=lambda item: item[1], reverse=True):
    #    print('%s: %.3f' % (k,v))
    #save(surf_strr[:-1]+'_.npy', np.array(torch.cat(hx).unsqueeze(0)))
    #save(surf_strr[:-1]+'_surf_strs.npy', surf_strs)


    # surf2pos probing
    # last_state_vector: (1024)
    # last_state_vector = last_states[0,0,:].cpu().detach()
    # z_vector = z[0,0,:].cpu().detach()
    # sft = nn.Softmax(dim=2)
    # (1, batchsize, vocab_size)
    # output_logits = model.probe_linear(last_states)
    # pred_tense = torch.argmax(sft(output_logits), 2)
    
    # visualize vector projections on predicted weight vectors
    # pos_weight_vector = model.probe_linear.weight[pred_pos.item()].cpu().detach()
    # vp = vector_projection(last_state_vector, pos_weight_vector)
    # x.append(torch.tensor(vp).unsqueeze(0))
   
    # to visualize vector projections on weight vectors
    # x.append(output_logits.squeeze(0).cpu().detach())
    #hx.append(last_state_vector.unsqueeze(0).cpu().detach())
    #zx.append(z_vector.unsqueeze(0).cpu().detach())

    # posdata.append(pos.item())
    # polardata.append(polar.item())
    # tensedata.append(tense.item())

    # predposdata.append(pred_pos.item())
    # predpolardata.append(pred_polar.item())
    # predtensedata.append(pred_tense.item())
    # root_str = ''.join(surface_vocab.decode_sentence(root.t())[1:-1])
    # surf_str = ''.join(surface_vocab.decode_sentence(surf.t())[1:-1]) 
    # feat_str = ''.join(feature_vocab.decode_sentence(feat.t())[1:-1]) 
    # root_strs.append(root_str)
    # surf_strs.append(surf_str)
    # feat_strs.append(feat_str)

ZX = np.array(torch.cat((zx))) 
print('number of ZX points: ', ZX.shape)
save('z_ae_original_turkish_test_data.npy', ZX)
save('surfs_ae_original_turkish_test_data.npy', surf_strs)

# X = np.array(torch.cat((x))) 
# print('number of points: ', X.shape)
#HX = np.array(torch.cat((hx))) 
#print('number of HX points: ', HX.shape)
#save('_rawbible_sigmorphon2021-task2-turkish-gold-data_ae_original_mu_states.npy', HX)
# save('sigmorphon2021-task2-turkish-gold-data_feat_strs.npy', feat_strs)
#save('_rawbible_sigmorphon2021-task2-turkish-gold-data_surf_strs.npy', surf_strs)

#ZX = np.array(torch.cat((zx))) 
#print('number of ZX points: ', ZX.shape)
#save('goldwords_sigmorphon2021-task2-turkish-gold-data_ae_original_mu_states.npy', ZX)
# save('sigmorphon2021-task2-turkish-gold-data_feat_strs.npy', feat_strs)
# save('goldwords_sigmorphon2021-task2-turkish-gold-data_surf_ae_strs.npy', surf_strs)

'''# (ninstances, hiddendim)
save('3814instances_FUT_ae_tense_projections_data.npy', X)
save('3814instances_FUT_ae_last_states_data.npy', HX)
save('3814instances_FUT_ae_feat_strs.npy', feat_strs)
save('3814instances_FUT_ae_surf_strs.npy', surf_strs)
save('3814instances_FUT_ae_tensedata.npy', tensedata)
save('3814instances_FUT_ae_predtensedata.npy', predtensedata)'''



## visualizations
'''
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