import json

'''# train maxll and change initial morphemes 
mbook    = 'morphs.txt' #'evaluation/morph_segmentation/results/vae/vae_5/avg/nsamples15000/word_given/prev_mid_next/eps0.0/30k_segments.txt'
testfile = 'evaluation/morph_segmentation/data/top40k_wordlist.tur'
for i in range(5):
    iter_prob = 0
    morph_book = read_morph_book()
    # Segment via morphbook
    with open('morphs.txt', 'w') as fseg:
        with open(testfile, 'r') as reader:
            for line in reader:
                line = line.strip()
                word = line.replace(' ', '')
                word = word.split('\t')[0]
                morphs = []
                morphs, prob = viterbi_segment(word, morph_book)
                iter_prob += prob

                #if morphs[0] not in root_book:
                #    morphs = [morphs[0] + morphs[1]] +morphs[2:]

                # do not let 2-char roots
                if len(morphs[0]) == 2 and len(morphs) > 2:
                    _mergedroot = morphs[0] + morphs[1]
                    morphs = [_mergedroot] + morphs[2:]
                
                if len(morphs) == 0:
                    fseg.write(word+'\n')
                else:
                    fseg.write(str(' '.join(morphs)+'\n'))
    print('iter prob: %.3f' % (iter_prob/40000))'''

def viterbi_segment(text, P, debug=False):
    """Find the best segmentation of the string of characters, given the
    UnigramTextModel P."""
    # best[i] = best probability for text[0:i]
    # words[i] = best word ending at position i
    n = len(text)
    words = [''] + list(text)
    best = [1.0] + [0.0] * n
    ## Fill in the vectors best, words via dynamic programming
    for i in range(n+1):
        for j in range(0, i):
            if debug:
                print('\ni-j',(i,j))
            w = text[j:i]
            if debug:
                print('w: ', w)
            if w not in P:
                continue
                #if debug:
                #    print('%s not in P, setting to 0' % w)
                #P[w] = 0
            if len(w) == 1 or len(text[:j]) ==1:
                continue
                #P[w] = -1000
            
            if debug:
                print('P[%s]: %.2f' % (w, P[w]))
                print('best[%d]: %.2f' % (j, best[j]))
                print('best[%d]: %.2f' % (i, best[i]))
                print('checking for P[%s] * best[%d] >= best[%d]' % (w,j,i))
            if P[w] * best[i - len(w)] >= best[i]:
                if debug:
                    print(True)
                    print('best[%d] set to %.2f' % (i, P[w] * best[i - len(w)]))
                    print('words[%d] set to %s' % (i,w))
                best[i] = P[w] * best[i - len(w)]
                words[i] = w
            else:
                if debug:
                    print(False)
                    print('words[%d] stays as %s' % (i,words[i]))
            if debug:
                print(words)
    ## Now recover the sequence of best words
    sequence = []; i = len(words)-1
    while i > 0:
        sequence[0:0] = [words[i]]
        i = i - len(words[i])
    if debug:
        print('result: ', sequence)
        print('best prob. ', best[-1])
    
    ## Return sequence of best words and overall probability
    return sequence, best[-1]

def read_morph_book():
    # Create morph book and read initial segments
    initial_segments = dict()
    morph_book = dict()
    root_book = dict()
    print('reading morph book from...%s' % mbook) 
    with open(mbook, 'r') as reader:
        for line in reader:
            line = line.strip()
            morphs = line.split(' ')
            word = line.replace(' ', '')
            initial_segments[word] = morphs

            root = morphs[0]
            if root not in root_book:
                root_book[root] = 1
            else:
                root_book[root]+= 1

            for i,morph in enumerate(morphs):
                if morph not in morph_book:
                    morph_book[morph] = 1
                else:
                    morph_book[morph] += 1
    dropped_keys = []
    for key, value in morph_book.items():
        if len(key) == 2:
            if not ((key[1].lower() == 'a' or key[1].lower() == 'e' or key[1].lower() == 'i' or key[1].lower() == 'o' or key[1].lower() == 'u') 
            or (key[0].lower() == 'a' or key[0].lower() == 'e' or key[0].lower() == 'i' or key[0].lower() == 'o' or key[0].lower() == 'u')):
                dropped_keys.append(key)
    for key in dropped_keys:
        morph_book.pop(key)
        print('dropped %s' % key)
    return morph_book, root_book

def read_second_segments():
    # Create second segments
    second_segments = dict()
    with open('evaluation/morph_segmentation/results/vae/vae_5/avg/nsamples15000/word_given/prev_mid_next/eps0.0/segments.txt', 'r') as reader:
        for line in reader:
            line = line.strip()
            morphs = line.split(' ')
            word = line.replace(' ', '')
            second_segments[word] = morphs
    return second_segments


mbook    = 'evaluation/morph_segmentation/results/ae/ae_3/avg/word_given/prev_mid_next/eps0.0/40ksegments.txt' #'/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/results/vae/vae_5/avg/nsamples15000/word_given/prev_mid_next/eps0.0/30k_segments.txt'#'10k_refined.segments'#'/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/results/vae/vae_5/avg/nsamples15000/word_given/prev_mid_next/eps0.0/10k_segments.txt'
testfile = 'evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur'

morph_book, root_book = read_morph_book()
second_segments = read_second_segments()
iter_prob = 0
# Segment via morphbook
with open('evaluation/morph_segmentation/results/viterbi_ae.segments', 'w') as fseg:
    with open(testfile, 'r') as reader:
        for line in reader:
            line = line.strip()
            word = line.replace(' ', '')
            word = word.split('\t')[0]
            morphs = []
            morphs, prob = viterbi_segment(word, morph_book, debug=False)
            iter_prob += prob

            # do not let 1-char roots
            if len(morphs[0]) == 1 and len(morphs) > 2:
                _mergedroot = morphs[0] + morphs[1]
                morphs = [_mergedroot] + morphs[2:]

            # do not let 2-char roots
            if len(morphs[0]) == 2 and len(morphs) > 2:
                _mergedroot = morphs[0] + morphs[1]
                morphs = [_mergedroot] + morphs[2:]
            
            # do not let unrecognized roots - only one check
            if (morphs[0] not in root_book) and (len(morphs)>1):
                morphs = [morphs[0] + morphs[1]] +morphs[2:]
           
            # # recover segments from another method
            # if len(morphs) == len(second_segments[word]):
            #     morphs = morphs
            # else:
            #     morphs = second_segments[word]

            # more_than_one_flag = 0
            # for i in morphs:
            #     if len(i) ==1:
            #         more_than_one_flag +=1
            # if more_than_one_flag > 1:
            #     morphs = second_segments[word]
            
            if len(morphs) == 0:
                fseg.write(word+'\n')
            else:
                fseg.write(str(' '.join(morphs)+'\n'))

