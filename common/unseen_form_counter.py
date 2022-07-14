from collections import defaultdict


seen_words = defaultdict(lambda: 0)
lang= 'turkish'
unseen_reinflects = defaultdict(lambda: 0)
unseen_infs = defaultdict(lambda: 0)

with open('/home/mugekural/dev/git/trmor/data/sigmorphon2016/turkish_zhou_merged','r') as reader:
    for line in reader:
        seen_words[line.strip()]+=1

with open('/home/mugekural/dev/git/trmor/data/sigmorphon2016/turkish-task3-test','r') as reader:
    testsize = 0
    for line in reader:
        inf_form = line.strip().split('\t')[0]
        reinf_form = line.strip().split('\t')[-1]

        if reinf_form not in seen_words:
            unseen_reinflects[reinf_form]+=1
        
        if inf_form not in seen_words:
            unseen_infs[inf_form]+=1
        
        testsize +=1

print('%d reinfs never seen' % len(unseen_reinflects))
print('%d infs never seen' % len(unseen_infs))
