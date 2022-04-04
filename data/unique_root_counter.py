input_file = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vqvae/data/sosimple.new.seenroots.val.txt'
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
