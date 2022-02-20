# -----------------------------------------------------------
# Date:        2022/02/19 
# Author:      Muge Kural
# Description: Visualizations of final hidden states (points) of trained model with t-SNE and colorizes a point based on its polarity label 
# -----------------------------------------------------------

from collections import defaultdict
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
from numpy import dot, save
from numpy.linalg import norm

import sys, argparse, random, torch, json, matplotlib, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from common.utils import *
from data.data import build_data
from model.charlm.charlm import CharLM
from common.vocab import VocabEntry
#matplotlib.use('Agg')

# annotation on hovers
def update_annot(ind, params):
    fig, annot, ax, sc, words = params
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([words[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event, params):
    fig, annot, ax, sc, words = params
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind, params)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

def config():
     # CONFIG
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.device = 'cuda'
    model_id = 'charlm_4'
    model_path, model_vocab  = get_model_info(model_id)
    # workaround to only get polar_vocab
    probe_id = 'charlm_4_probe_polar'
    _, (_, polar_vocab)  = get_model_info(probe_id) 

    # logging
    args.logdir = 'evaluation/visualization/tsne/label_colorized/polarity/results/'+model_id+'/'
    args.figfile   = args.logdir +'vis.png'
    try:
        os.makedirs(args.logdir)
        print("Directory " , args.logdir ,  " Created ") 
    except FileExistsError:
        print("Directory " , args.logdir ,  " already exists")
    # initialize model
    # load vocab (to initialize the model with correct vocabsize)
    with open(model_vocab) as f:
        word2id = json.load(f)
        args.surf_vocab = VocabEntry(word2id)
    # load vocab (to initialize the model with correct vocabsize)
    with open(polar_vocab) as f:
        word2id = json.load(f)
        args.polar_vocab = VocabEntry(word2id)
    args.vocab = (args.surf_vocab, args.polar_vocab)

    # colors
    with open('evaluation/visualization/tsne/label_colorized/colors.json', 'r') as f:
        args.color_dict = json.load(f)
  
    # model
    model_init = uniform_initializer(0.01); emb_init = uniform_initializer(0.1)
    args.ni = 512; #for ae,vae,charlm
    args.nz = 32   #for ae,vae
    args.enc_nh = 1024; args.dec_nh = 1024;  #for ae,vae
    args.nh = 1024 #for ae,vae,charlm
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0 #for ae,vae,charlm
    args.dec_dropout_in = 0.0; args.dec_dropout_out = 0.0 #for ae,vae
    args.model = CharLM(args, args.surf_vocab, model_init, emb_init)
    # load model weights
    args.model.load_state_dict(torch.load(model_path))
    args.model.to(args.device)
    args.model.eval()
    # data
    args.tstdata = 'evaluation/visualization/tsne/label_colorized/polarity/data/surf.uniquesurfs.trn.txt'
    args.maxtstsize = 10000
    args.batch_size = 1
    return args


def main():
    args = config()
    data, batches = build_data(args)
    fhs_vectors = []; words = []; label_colors =[]
    with torch.no_grad():
        # loop through each instance
        for data in batches:
            surf, polar = data
            # fhs: (1,1,nh)
            fhs, _ = args.model(surf)
            fhs_vectors.append(fhs)
            word =''.join(args.surf_vocab.decode_sentence(surf[0][1:-1]))
            words.append(word)
            label_colors.append( args.color_dict[str(polar.item())])
            

    # (numinstances, nh)
    fhs_tensor = torch.stack(fhs_vectors).squeeze(1).squeeze(1).cpu()
    tsne_results = TSNE(n_components=2, verbose=1).fit_transform(fhs_tensor)
    fig,ax = plt.subplots()
    sc = plt.scatter(tsne_results[:,0], tsne_results[:,1], color=label_colors)
    plt.savefig(args.figfile)
    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", lambda event: hover(event, [fig, annot, ax, sc, words]))
    plt.show()




if __name__=="__main__":
    main()




'''#t-sne config

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
    surf_str = surface_vocab.decode_sentence(surf.t()) 
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
        print('%s similarity: %.3f' % (k,v))
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
    # feat_strs.append(feat_str)'''



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
'''