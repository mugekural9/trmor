import json

def make_children(numentry, known_children=[]):
  children = []
  for e in range(numentry):
    children.append({"name": str(e),
    "children": known_children
    })
  return children



num_dict_entries = [5,5,5]
#c   = make_children(num_dict_entries[5])
#c   = make_children(num_dict_entries[4], c)
#c   = make_children(num_dict_entries[3], c)
c   = make_children(num_dict_entries[2])
c   = make_children(num_dict_entries[1], c)
c_3 = make_children(num_dict_entries[0], c)
with open('template_3x5.json', 'w') as w:
  _data = json.dumps(c_3)
  w.write(_data)


begin = {}
begin['name'] = "titre"
begin['children'] = [{"name": "0", "children": [{"name": "0", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "1", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "2", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "3", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "4", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}]}, {"name": "1", "children": [{"name": "0", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "1", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "2", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "3", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "4", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}]}, {"name": "2", "children": [{"name": "0", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "1", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "2", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "3", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "4", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}]}, {"name": "3", "children": [{"name": "0", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "1", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "2", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "3", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "4", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}]}, {"name": "4", "children": [{"name": "0", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "1", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "2", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "3", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}, {"name": "4", "children": [{"name": "0", "children": []}, {"name": "1", "children": []}, {"name": "2", "children": []}, {"name": "3", "children": []}, {"name": "4", "children": []}]}]}]

with open('model/vqvae/results/generation/vqvae_3x5/samples.txt', 'r') as reader:
  for line in reader:
    line = line.strip()
    val, key = line.split('\t')
    i,m,k = key[1:-1].strip().split(',')[1:]
    i,m,k= int(i), int(m), int(k)
    begin['children'][i]['children'][m]['children'][k]['children'].append({"name":val})

with open('titre_3x5.json', 'w') as f:
  newdata = json.dumps(begin, ensure_ascii=False)
  f.write(newdata)

