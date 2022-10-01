unsup_words = dict()
sup_words = dict()
with open('/home/mugekural/dev/git/trmor/data/sigmorphon2016/turkish_zhou_merged','r') as reader:
    for line in reader:
        unsup_words[line.strip()] = 1


with open('/home/mugekural/dev/git/trmor/data/sigmorphon2016/turkish-task3-train','r') as reader:
    for line in reader:
        sup_words[line.strip().split('\t')[0]] = 1

unique_words =[]
for key,val in sup_words.items():
    if key not in unsup_words:
        unique_words.append(key)

with open('/home/mugekural/dev/git/trmor/data/sigmorphon2016/turkish_zhou_merged_unique_without_lxsrc','w') as writer:
    for word in unique_words:
        writer.write(word+'\n')
