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

'''# POS (root)
input = open("pos(surf)/surfpos.uniquesurfs.trn.txt", "r")
outputs = dict()
outputs['Verb']= open("pos(root)_verb.uniquesurfs.txt", "w")
outputs['Noun']= open("pos(root)_noun.uniquesurfs.txt", "w")
outputs['Adj'] = open("pos(root)_adj.uniquesurfs.txt",  "w")
keywords = ['Verb', 'Noun', 'Adj']
surfs  = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  if tags[1] == 'Noun' or tags[1] == 'Verb' or tags[1] == 'Adj':
    surfs[surf] = line

# create roots & frequencies dict
roots  = dict()
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
with open('roots_and_frequencies.json', 'w') as file:
  file.write(json.dumps(sorted(roots_and_frequencies.items(), key=lambda item: len(item[1]), reverse=True), ensure_ascii=False))
disamb = 0
for key, value in roots_and_frequencies.items():
  if len(value) > 1:
    disamb += 1
print('disamb: ', disamb) 
for root, line in roots.items():
  _, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  root, root_postag = tags[0], tags[1]
  if root_postag in keywords:
    outputs[root_postag].write(line)
input.close()'''

'''# Tense
input = open("surf/surf.uniquesurfs.trn.txt", "r")
outputs = dict()
outputs['Past']= open("tense_past.uniquesurfs.txt", "w")
outputs['Narr']= open("tense_narr.uniquesurfs.txt", "w")
outputs['Fut']= open("tense_fut.uniquesurfs.txt", "w")
outputs['Prog1']= open("tense_prog1.uniquesurfs.txt", "w")
outputs['Prog2']= open("tense_prog2.uniquesurfs.txt", "w")
outputs['Aor']= open("tense_aor.uniquesurfs.txt", "w")
outputs['Cop']= open("tense_cop.uniquesurfs.txt", "w")
keywords = ['Past', 'Narr', 'Fut', 'Prog1', 'Prog2', 'Aor', 'Cop']
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  #tags = line.strip().split('+') # does this cause a problem?
  for tag in reversed(tags):
      if tag in keywords:
          outputs[tag].write(line)
          break
input.close()'''

'''# Polarity
input = open("trmor2018.uniquesurfs.txt", "r")
outputs = dict()
outputs['Pos']= open("polar_pos.uniquesurfs.txt", "w")
outputs['Neg']= open("polar_neg.uniquesurfs.txt", "w")
keywords = ['Pos', 'Neg']
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  for tag in reversed(tags):
      if tag in keywords:
          outputs[tag].write(line)
          break
input.close()'''


'''# Polarity Info Existence
input = open("trmor2018.uniquesurfs.txt", "r")
output_polar = open("polar.uniquesurfs.txt", "w")
output_nopolar = open("nopolar.uniquesurfs.txt", "w")
keywords = ['Pos', 'Neg']
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  polar = False
  for tag in reversed(tags):
      if tag in keywords:
          polar = True
          break
  if polar:
    output_polar.write(line)
  else:
    output_nopolar.write(line)
input.close()'''


'''# Surf POS
input = open("trmor2018.uniquesurfs.txt", "r")
outputs = dict()
outputs['Verb']= open("surfpos_verb.uniquesurfs.txt", "w")
outputs['Noun']= open("surfpos_noun.uniquesurfs.txt", "w")
outputs['Adj'] = open("surfpos_adj.uniquesurfs.txt",  "w")
outputs['Adverb'] = open("surfpos_adverb.uniquesurfs.txt",  "w")
outputs['Pron'] = open("surfpos_pron.uniquesurfs.txt",  "w")
outputs['Postp'] = open("surfpos_postp.uniquesurfs.txt",  "w")
outputs['Num'] = open("surfpos_num.uniquesurfs.txt",  "w")
outputs['Det'] = open("surfpos_det.uniquesurfs.txt",  "w")
outputs['Conj'] = open("surfpos_conj.uniquesurfs.txt",  "w")
outputs['Interj'] = open("surfpos_interj.uniquesurfs.txt",  "w")
outputs['Ques'] = open("surfpos_ques.uniquesurfs.txt",  "w")
outputs['Dup'] = open("surfpos_dup.uniquesurfs.txt",  "w")
outputs['Punc'] = open("surfpos_punc.uniquesurfs.txt",  "w")
keywords = ['Verb', 'Noun', 'Adj', 'Adverb', 'Pron', 'Postp', 'Num', 'Det', 'Conj', 'Interj', 'Ques', 'Dup', 'Punc']
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  for tag in reversed(tags):
      if tag in keywords:
          outputs[tag].write(line)
          break'''


'''# words with no suffix 
input = open("trmor2018.uniquesurfs.txt", "r")
output = open("trmor2018.nosuffixwords.txt", "w")
nosuffixsurfs  = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  if surf == tags[0]:
    nosuffixsurfs[surf] = line
    
for surf, line in nosuffixsurfs.items():
  output.write(line)
input.close()'''


'''# Remove particular set from data
input = open("tense/tense_aor.uniquesurfs.txt", "r")
output = open("simple.tense_aor.uniquesurfs.txt", "w")
keywords = ['Past', 'Narr', 'Fut', 'Prog1', 'Aor']
for line in input:
  exists = False
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  for tag in reversed(tags):
    if 'DB' in tags:
      break
    if tag in keywords:
      if tag != 'Aor':
        break
      else: 
        exists = True
        break
  if exists:
    output.write(line)
input.close()'''

'''# Remove particular set from data
input = open("simpleshot/simple.tense_aor.uniquesurfs.txt", "r")
output = open("sosimple.tense_aor.uniquesurfs.txt", "w")
keywords = ['Past', 'Narr', 'Fut', 'Prog1', 'Aor', 'Cond', 'Desr', 'Neces', 'Prog2', 'Opt', 'Imp']
for line in input:
  multiple_tense = False
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  for tag in reversed(tags):
    if tag in keywords and tag != 'Aor':
      multiple_tense = True
  if not multiple_tense:
    output.write(line)
input.close()'''





'''# (root frequencies)
input = open("sosimpleshot/sosimple.tense.uniquesurfs.trn.txt", "r")
roots_and_frequencies_trn = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  root = tags[0]
  tense = tags[-2]
  if root not in roots_and_frequencies_trn.keys():
    roots_and_frequencies_trn[root] = dict()
  if tense not in roots_and_frequencies_trn[root]:
    roots_and_frequencies_trn[root][tense] = 0
  roots_and_frequencies_trn[root][tense] += 1

with open('tense_roots_and_frequencies.tst.json', 'w') as file:
  file.write(json.dumps(sorted(roots_and_frequencies.items(), key=lambda item: len(item[1]), reverse=True), ensure_ascii=False))
disamb = 0
for key, value in roots_and_frequencies.items():
  if len(value) > 1:
    disamb += 1
print('disamb: ', disamb) 
print(len(roots_and_frequencies))

input = open("sosimpleshot/sosimple.tense.uniquesurfs.tst.txt", "r")
roots_and_frequencies_tst = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  root = tags[0]
  tense = tags[-2]
  if root not in roots_and_frequencies_tst.keys():
    roots_and_frequencies_tst[root] = dict()
  if tense not in roots_and_frequencies_tst[root]:
    roots_and_frequencies_tst[root][tense] = 0
  roots_and_frequencies_tst[root][tense] += 1

inputseen = open("sosimpleshot/sosimple.seenroots_duringtrn.uniquesurfs.tst.txt", "w")
inputunseen = open("sosimpleshot/sosimple.unseenroots_duringtrn.uniquesurfs.tst.txt", "w")
seen_roots = []
unseen_roots = []
for key in roots_and_frequencies_tst.keys():
  if key in roots_and_frequencies_trn.keys():
    seen_roots.append(key)
  else:
    unseen_roots.append(key)

for root in seen_roots:
  inputseen.write(root+'\n')
for root in unseen_roots:
  inputunseen.write(root+'\n')'''

# Words with unique roots
input = open("surf/surf.uniquesurfs.tst.txt", "r")
roots_and_frequencies = dict()
for line in input:
  surf, tags = line.strip().split('\t')
  tags = tags.replace('^', '+').split('+')
  root = tags[0]
  if root not in roots_and_frequencies.keys():
    roots_and_frequencies[root] = 1
  else:
    roots_and_frequencies[root] += 1
with open('roots_and_frequencies.tst.json', 'w') as file:
  file.write(json.dumps(dict(sorted(roots_and_frequencies.items(), key=lambda item: item[1], reverse=True)), ensure_ascii=False))

'''output_trn = open("uniqueroots.trn.txt", "a")
output_val = open("uniqueroots.val.txt", "a")
output_tst = open("uniqueroots.tst.txt", "a")
trnsize = 500
valsize = 50
tstsize = 100
input1 = open("uniqueroot/pos(root)_verb.uniqueroots.txt",  "r")

inputs = [input1]

for input in inputs:
  c = 0
  for line in input:
    if c < trnsize:
      output_trn.write(line)
    elif c < trnsize + valsize:
      output_val.write(line)
    elif c < trnsize + valsize + tstsize:
      output_tst.write(line)
    else:
      break
    c += 1'''
