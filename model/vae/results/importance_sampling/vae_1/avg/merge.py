import re

words =[]
f = open("../../../../data/goldstdsample.tur", "r")
i=1
for line in f:
  if i>5:
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
      f  = open("nsamples1-20000/subword_given/importance_sampling.txt_"+word, "r")
      f2 = open("nsamples20000-30000/subword_given/importance_sampling.txt_"+word, "r")
      #f3 = open("nsamples30000-35000/importance_sampling_muge.txt_"+word, "r")
      #f4 = open("nsamples40000-50000/importance_sampling_muge.txt_"+word, "r")
    except:
      continue
    for line in f:
      if 'mean' in line:
        mean =  float(re.search('mean:(.*),', line).group(1))
        print(mean)
        o.write('%.4f\n' % (mean))
    for line in f2:
      if 'mean' in line:
        mean =  float(re.search('mean:(.*),', line).group(1))
        print(mean)
        o.write('%.4f\n' % (mean))

