import re

words =[]
f = open("goldstd_trainset.segments.tur", "r")
i=1
for line in f:
  if False:#i>5:
    break
  else:
    i+=1
    words.append(line.split('\t')[0])

o = open("out", "w")
for _word in words:
  for i in range(1,len(_word)+1):
    word = _word[:i]
    print('\n'+word)
    o.write('\n')
    o.write(word+'\n')

    try:
      f1 = open("avg/nsamples1-60000/subword_given/importance_sampling.txt_"+word, "r")
    except:
      continue

    for line in f1:
      if 'mean' in line:
        mean =  float(re.search('mean:(.*),', line).group(1))
        print(mean)
        o.write('%.4f\n' % (mean))
