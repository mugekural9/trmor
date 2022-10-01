lines = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/data/goldstd_mc05-10aggregated.segments.tur") as reader:
    for line in reader:
        word = line.split('\t')[0]
        lines[word] = line

with open("/kuacc/users/mugekural/workfolder/dev/git/trmor/evaluation/morph_segmentation/results/vae/VAE_FINAL/avg/nsamples32000/word_given/prev_mid_next/eps0.0/segments.txt") as reader:
    with open('sorted.txt','w') as writer:
        for line in reader:
            word = line.replace(' ','').strip()
            writer.write(lines[word])
