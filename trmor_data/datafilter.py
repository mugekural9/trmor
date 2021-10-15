input = open("trmor2018.filtered", "r")
output = open("tense_pres.txt", "w")

for line in input:
  if 'Pres+A' in line:
    output.write(line)

input.close()
output.close()