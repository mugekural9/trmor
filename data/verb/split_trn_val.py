import json
pos_tags_dict = dict()
pos_tags = ['Adj', 'Adverb', 'Verb', 'Noun', 'Conj', 'Det', 'Dup', 'Interj', 'Num', 'Postp', 'Pron', 'Ques']
for tg in pos_tags:
    pos_tags_dict [tg] = []
    
with open('../trmor2018.uniquesurfs.txt', 'r') as reader:
    count = 0
    for line in reader: 
        count += 1
        split_line = line.split()
        tags     = split_line[1].replace('^','+').split('+')[1:]
        postag_info_exists = False
        for tag in reversed(tags):
            if tag in pos_tags:
                #pos_tags_dict[tag] += 1
                pos_tags_dict[tag].append(line)
                postag_info_exists = True
                break
        if not postag_info_exists:
            print('NO POS TAG info')

trnsize = 0
valsize = 0
trndata = []; valdata = []
roots =  dict()

for line in pos_tags_dict['Verb']:
    split_line = line.split()
    tags     = split_line[1].replace('^','+').split('+')
    root = tags[0]
    unsplit_line = '\t'.join(split_line)
    if root not in roots:
        roots[root] = 1
        trndata.append(unsplit_line)
    else:
        valdata.append(unsplit_line)
        roots[root] += 1

print('Unique roots in trn:...', len(trndata))
print('All seen roots in val:...', len(valdata))
print('Now lets move a part of val to trn')

trndata = trndata+valdata[:7995] # make it 10k 
valdata = valdata[7995:(7995+3000)] # make it 3k

with open('trmor2018.uniquesurfs.verbs.uniquerooted.trn.txt', 'w') as writer:
    for line in trndata:
        writer.write(line+'\n')


with open('trmor2018.uniquesurfs.verbs.seenroots.val.txt', 'w') as writer:
    for line in valdata:
        writer.write(line+'\n')
    