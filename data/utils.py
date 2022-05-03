import json, random

def unique_root_counter(input_file):
    #input_file = 'model/vqvae/data/sosimple.new.seenroots.val.txt'
    roots_dict = dict()
    with open(input_file, 'r') as reader:
        count = 0
        for line in reader: 
            split_line = line.split()
            tags     = split_line[1].replace('^','+').split('+')
            root = tags[0]
            if root not in roots_dict:
                roots_dict[root] =  1
            else:
                roots_dict[root] += 1
    print(len(roots_dict))


def suffix_counter(input_file):
    #input_file = 'data/noun/trmor2018.uniquesurfs.nouns.uniquerooted.trn.txt'
    no_suffix = []; with_suffix = []
    with open(input_file, 'r') as reader:
        count = 0
        for line in reader: 
            split_line = line.split()
            tags     = split_line[1].replace('^','+').split('+')
            root = tags[0]
            if split_line[0] == root:
                no_suffix.append(line)
            else:
                with_suffix.append(line)

    print('no suffix: ')   
    print(len(no_suffix))
    #print(no_suffix)

    print('with suffix: ')   
    print(len(with_suffix))
    #print(with_suffix)

    with open('data/noun/trmor2018.uniquesurfs.nouns.uniquerooted.trn.nosuffix.txt', 'w') as writer:
        for line in no_suffix:
            writer.write(line)

    with open('data/noun/trmor2018.uniquesurfs.nouns.uniquerooted.trn.withsuffix.txt', 'w') as writer:
        for line in with_suffix:
            writer.write(line)


def filter_with_pos():
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


def unique_roots():
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


def split_trn_val():
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
        
        
def generate_with_root():
    unique_roots = dict()
    with open('test.txt', 'r') as reader:
        for line in reader:
            split_line = line.split('\t')
            tags     = split_line[1].replace('^','+').split('+')
            root = tags[0]
            if root not in unique_roots:
                unique_roots[root] = 1
            else:
                unique_roots[root] += 1

    sorted_roots =  dict(sorted(unique_roots.items(), key=lambda item: item[1]))

    top50_roots = list(sorted_roots.keys())[-50:]

    lines = []
    with open('test.txt', 'r') as reader:
        for line in reader:
            split_line = line.split('\t')
            tags     = split_line[1].replace('^','+').split('+')
            root = tags[0]
            if root in top50_roots:
                lines.append(line)


    random.shuffle(lines)
    with open('1000verbs.simple.trn.txt', 'w') as f:
        for line in lines[:1000]:
            f.write(line)

    with open('1000verbs.simple.val.txt', 'w') as f:
        for line in lines[1000:]:
            f.write(line)