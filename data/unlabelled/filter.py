with open('data/unlabelled/wordlist.tur', 'r') as reader:
    words = dict()
    for line in reader:
        freq, word = line.strip().split(' ')
        words[word] = int(freq)
         
    ordered_words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
    #top 50k words
    lwds = list(ordered_words.items())[-53000:-50000]
    lwds = dict(lwds)

with open('data/unlabelled/theval.tur', 'w') as f:
    for k,v in lwds.items():
        f.write(str(v)+' '+k+'\n')
        #f.write(k+'\n')
