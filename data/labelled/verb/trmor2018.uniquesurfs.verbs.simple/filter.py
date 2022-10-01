c=0
words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/trmor/data/labelled/verb/trmor2018.uniquesurfs.verbs.simple/trmor2018.uniquesurfs.verbs.simple.txt", "r") as reader:
    i=0
    for line in reader:
        i+=1
        line = line.strip()
        surf,tags = line.split('\t')
        tense = tags.split("+")[-2]
        if not (tense=='Aor' or tense=='Prog1' or tense=='Fut' or tense=='Narr' or tense=='Past'):
            c+=1         
        else:
            words[surf] = 1
print(i)
print(c)
print(i-c)
print(len(words))
