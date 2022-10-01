tag_key = 'per'

with open('/kuacc/users/mugekural/workfolder/dev/git/trmor/data/sigmorphon2016/hungarian-task3-train', 'r') as reader:
    with open(tag_key+'.trn.txt','w') as writer:
        for line in reader:
            if tag_key in line:
                writer.write(line)


with open('/kuacc/users/mugekural/workfolder/dev/git/trmor/data/sigmorphon2016/hungarian-task3-test', 'r') as reader:
    with open(tag_key+'.val.txt','w') as writer:
        for line in reader:
            if tag_key in line:
                writer.write(line)