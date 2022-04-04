input = open("sosimple.new.trn.combined.txt", "r")
#input2 = open("sosimple.new.val.combined.txt", "r")
output = open("sosimple.new.trn.combined.100unique_roots.txt", "w")

seen_roots = dict()
for line in input:
    if len(seen_roots) == 100:
        break
    output.write(line)
    root = line.split('\t')[1].split('+')[0]
    if root not in seen_roots:
        seen_roots[root] = 1
    else:
        seen_roots[root] +=1

print(len(seen_roots))
#for line in input2:
#    root = line.split('\t')[1].split('+')[0]
#    if root in seen_roots:
#        output.write(line)
