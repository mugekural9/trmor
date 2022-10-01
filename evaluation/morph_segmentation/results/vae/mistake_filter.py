from torch import equal


suggested = 'evaluation/morph_segmentation/results/vae/VAE_FINAL/avg/nsamples32000/subword_given/prev_mid_next/eps0.0/segments.txt'
gold = '/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur'

suggested_segs = dict()
with open(suggested,'r') as sreader:
    for line in sreader:
        line = line.strip()
        morphs = line.split(' ')
        word = line.replace(' ', '')
        suggested_segs[word] = morphs

gold_segs = dict()
gold_segs_line = dict()
with open(gold,'r') as sreader:
    for line in sreader:
        line = line.strip()
        word, _morphs = line.split('\t')
        gold_segs_line[word] = _morphs
        gold_segs[word] = []
        if ',' in _morphs:
            _morphs = _morphs.split(',')
            for morphs in _morphs:
                morphs = morphs.strip()
                morphs = morphs.split(' ')
                gold_segs[word].append(morphs) 
        else:
            morphs = _morphs.split(' ')
            gold_segs[word].append(morphs) 

under=0
over =0
equa=0

mistakes = open('subwordgiven_mistakes.txt','w')
never_splitted = open('subwordgiven_neversplitted.txt','w')
unders = open('subwordgiven_unders.txt','w')
overs = open('subwordgiven_overs.txt','w')
equals =  open('subwordgiven_equals.txt','w')
for key,val in suggested_segs.items():
    if val not in gold_segs[key]:
        mistakes.write(key+'\t'+ gold_segs_line[key]+'\t'+str(val)+'\n')
        if len(val) < min([len(g) for g in gold_segs[key]]):
            under+=1
            if len(val) == 1:
                never_splitted.write(key+'\t'+ gold_segs_line[key]+'\t'+str(val)+'\n')
            else:
                unders.write(key+'\t'+ gold_segs_line[key]+'\t'+str(val)+'\n')
        if len(val) > min([len(g) for g in gold_segs[key]]):
            overs.write(key+'\t'+ gold_segs_line[key]+'\t'+str(val)+'\n')
            over+=1
        if len(val) == min([len(g) for g in gold_segs[key]]):
            equals.write(key+'\t'+ gold_segs_line[key]+'\t'+str(val)+'\n')
            equa+=1
print(under)
print(over)
print(equa)