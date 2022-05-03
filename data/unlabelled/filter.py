with open('data/morph_segmentation/wordlist.tur', 'r') as reader:
    words = dict()
    for line in reader:
        freq, word = line.strip().split(' ')
        words[word] = int(freq)
         
    ordered_words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1])}
    #top 50k words
    lwds = list(ordered_words.items())[-50000:]
    lwds = dict(lwds)

with open('data/morph_segmentation/top50k_wordlist.tur', 'w') as f:
    for k,v in lwds.items():
        f.write(str(v)+' '+k+'\n')
        #f.write(k+'\n')
