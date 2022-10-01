import json
with open("TRN.surfs.json") as trn_reader:
    trn_lemmas = json.load(trn_reader)
with open("VAL.surfs.json") as val_reader:
    val_lemmas = json.load(val_reader)

seen = 0
for k,v in val_lemmas.items():
    if k in trn_lemmas:
        seen +=v
print(seen, 'over ', 1597)