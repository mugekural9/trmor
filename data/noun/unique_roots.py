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

sorted_roots = dict(sorted(roots.items(), key=lambda item: item[1], reverse=True))
with open("unique_verb_roots.json", "w") as outfile:
    json.dump(sorted_roots, outfile, ensure_ascii=False)

