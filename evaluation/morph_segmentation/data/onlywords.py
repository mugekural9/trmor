onlywords = open("onlywords.txt", "w")
with open("/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur") as reader:
    for line in reader:
        word = line.split('\t')[0]
        onlywords.write(word+'\n')