import json
from collections import defaultdict

model_id = '2x30_suffixd512'
   

gold_tags_dict = dict(); found_tags_dict = dict(); found_tags_dict_r = dict()
with open('data/sigmorphon2016/turkish-task2-train', 'r') as reader:
  for line in reader:
    split_line = line.split('\t')
    tag  = split_line[0]
    word = split_line[1]
    gold_tags_dict[word] = tag

#for  c in list(range(8)):
if True:
  c=2
  id_tags = dict()
  with open('model/vqvae/results/training/12229_instances/'+model_id+'/'+str(c)+'_cluster.json', 'r') as f:
    modeldata = json.load(f)
  tag_counts=defaultdict(lambda: 0)
  for key,values in modeldata.items():
    for value in values:
      instance =value[3:-4]
      tag = gold_tags_dict[instance]
      tag_counts[tag] += 1
      if key not in id_tags:
        id_tags[key]=defaultdict(lambda:0)
      id_tags[key][tag] +=1
    for value in values:  
      instance =value[3:-4]
      tag = gold_tags_dict[instance]
      if tag not in found_tags_dict_r:
        found_tags_dict_r[tag] = dict()
      if key not in found_tags_dict_r[tag]:
        found_tags_dict_r[tag][key] = 0
      found_tags_dict_r[tag][key] +=1
  for k,v in found_tags_dict_r.items():
      x = found_tags_dict_r[k]
      found_tags_dict_r[k] = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}

  with open(model_id+'_id_to_tags_'+str(c)+'.json', 'w') as f:
      f.write(json.dumps(id_tags))

  '''tag_counts =  dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
  with open('tag_counts.json', 'w') as f:
      f.write(json.dumps(tag_counts))'''


  with open(model_id+'_tags_to_id_'+str(c)+'.json', 'w') as f:
      f.write(json.dumps(found_tags_dict_r))

  with open('data/sigmorphon2016/turkish-task3-test', 'r') as reader:
    todo = []
    found_tags = 0; not_found_tags = 0
    for line in reader:
      split_line = line.split('\t')
      asked_tag  = split_line[1]
      inflected_word = split_line[0]
      reinflected_word = split_line[2].strip()
      if asked_tag not in found_tags_dict_r:
        not_found_tags+=1
        print(asked_tag)
      else:
        found_tags+=1
        todo.append((inflected_word +' ||| '+ str(found_tags_dict_r[asked_tag])+' ||| '+ asked_tag +' ||| '+ reinflected_word))

  print(found_tags,'  found.', not_found_tags,' not found.')
  with open(model_id+'_turkish-task3-test_todo_LSTM_'+str(c), 'w') as writer:
    for i in todo:
      writer.write(str(i)+'\n')
