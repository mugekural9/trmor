import json, collections
from collections import defaultdict

'''# SURF 
input = open("trmor2018.filtered", "r")
output = open("trmor2018.uniquesurfs.txt", "w")
surfs  = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  surfs[surf] = line
for surf, line in surfs.items():
  output.write(line)
input.close()'''

'''# POS
input = open("trmor2018.filtered", "r")
outputs = dict()
outputs['Verb']= open("pos_verb.txt", "w")
outputs['Noun']= open("pos_noun.txt", "w")
outputs['Adj'] = open("pos_adj.txt",  "w")
keywords = ['Verb', 'Noun', 'Adj']
surfs  = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  if tags[1] == 'Noun' or tags[1] == 'Verb' or tags[1] == 'Adj':
    surfs[surf] = line
roots  = dict()
counts = dict()
roots_and_frequencies = dict()
for surf, line in surfs.items():
  _, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  root, root_postag = tags[0], tags[1]
  roots[root] = line 
  if root not in roots_and_frequencies.keys():
    roots_and_frequencies[root] = dict()
  if root_postag not in roots_and_frequencies[root]:
    roots_and_frequencies[root][root_postag] = 0
  roots_and_frequencies[root][root_postag] += 1
  if root_postag not in counts:
    counts[root_postag] = 1
  else:
    counts[root_postag] += 1
  if root_postag in keywords:
    outputs[root_postag].write(line)
output = open("pos/pos.uniqueroots.txt",  "w")
for root, line in roots.items():
  output.write(line)
input.close()
with open('roots_and_frequencies.json', 'w') as file:
  file.write(json.dumps(sorted(roots_and_frequencies.items(), key=lambda item: len(item[1]), reverse=True), ensure_ascii=False))
disamb = 0
for key, value in roots_and_frequencies.items():
  if len(value) > 1:
    disamb += 1
print('disamb: ', disamb)''' 

'''# Tense
input = open("trmor2018.filtered", "r")
outputs = dict()
outputs['Past']= open("tense_past.txt", "w")
outputs['Narr']= open("tense_narr.txt", "w")
outputs['Fut']= open("tense_fut.txt", "w")
outputs['Prog1']= open("tense_prog1.txt", "w")
outputs['Prog2']= open("tense_prog2.txt", "w")
outputs['Aor']= open("tense_aor.txt", "w")
outputs['Cop']= open("tense_cop.txt", "w")
keywords = ['Past', 'Narr', 'Fut', 'Prog1', 'Prog2', 'Aor', 'Cop']
for line in input:
  tags = line.strip().split('+')
  for tag in reversed(tags):
      if tag in keywords:
          outputs[tag].write(line)
          break
input.close()'''

'''# Polarity
input = open("trmor2018.filtered", "r")
outputs = dict()
outputs['Pos']= open("polar_pos.txt", "w")
outputs['Neg']= open("polar_neg.txt", "w")
keywords = ['Pos', 'Neg']
for line in input:
  tags = line.strip().split('+')
  for tag in reversed(tags):
      if tag in keywords:
          outputs[tag].write(line)
          break
input.close()'''


output_trn = open("surf.trn.txt", "a")
output_val = open("surf.val.txt", "a")
trnsize = 50000
valsize = 6729
input = open("trmor2018.uniquesurfs.txt", "r")
c = 0
for line in input:
  if c < trnsize:
    output_trn.write(line)
  elif c < trnsize + valsize:
    output_val.write(line)
  else:
    break
  c += 1
