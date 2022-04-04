import json
pos_tags_dict = dict()
pos_tags = ['Adj', 'Adverb', 'Verb', 'Noun', 'Conj', 'Det', 'Dup', 'Interj', 'Num', 'Postp', 'Pron', 'Ques']
for tg in pos_tags:
    pos_tags_dict [tg] = []
    
with open('trmor2018.uniquesurfs.txt', 'r') as reader:
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


#change related textfile part and dict key
with open('trmor2018.uniquesurfs.nouns.txt', 'w') as writer:
    roots =  dict()
    for line in pos_tags_dict['Noun']:
        split_line = line.split()
        writer.write(line)

