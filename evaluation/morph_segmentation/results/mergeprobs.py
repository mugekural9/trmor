import json
with open('evaluation/morph_segmentation/results/vae/vae_1/sum/nsamples15000/word_given/prev_mid_next/eps0.0/probs.json', 'r') as json_file:
    subword_logps = json.load(json_file)
    
with open('evaluation/morph_segmentation/results/vae/vae_1/avg/nsamples15000/word_given/prev_mid_next/eps0.0/probs.json', 'r') as json_file:
    word_logps = json.load(json_file)

merged_logps = dict()
for k,v in subword_logps.items():
    for k2, v2 in subword_logps[k].items():
        merged_logps[k2] =  0.2 * v2 + 0.8 * word_logps[k][k2]
        subword_logps[k][k2] =  merged_logps[k2]

with open('merged_40000_8.2.json', 'w') as json_file:
    json_object = json.dumps(subword_logps, indent = 4)
    json_file.write(json_object)