import random
lines = []
with open('trn_4x10.txt', 'r') as reader:
    for line in reader:
        lines.append(line)

random.shuffle(lines)

with open('trn_4x10_shuffled.txt', 'w') as writer:
    for line in lines[:10000]:
        writer.write(line)

with open('val_4x10_shuffled.txt', 'w') as writer:
    for line in lines[10000:]:
        writer.write(line)