import random

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