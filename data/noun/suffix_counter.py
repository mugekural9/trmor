input_file = 'data/noun/trmor2018.uniquesurfs.nouns.uniquerooted.trn.txt'
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