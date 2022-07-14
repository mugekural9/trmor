from cgi import test
from collections import defaultdict


lang= 'turkish'
model_id ='semisup_batchsize128_beta_0.2_11x6_bi_kl_0.2_epc190'


def counter(lang, model_id):
    dict_combs = defaultdict(lambda: 0)
    unseen_combs = defaultdict(lambda: 0)

    with open('model/vqvae/results/analysis/'+lang+'/'+model_id+'/train_'+model_id+'_shuffled.txt','r') as reader:
        for line in reader:
            dict_combs[line.split('\t')[-2]] +=1

    with open('model/vqvae/results/analysis/'+lang+'/'+model_id+'/test_'+model_id+'_shuffled.txt','r') as reader:
        testsize = 0
        for line in reader:
            testsize +=1
            comb= line.split('\t')[-2]
            if comb not in dict_combs:
                unseen_combs[comb] +=1

    print('model: %s' % model_id)
    print('%d combs never seen' % len(unseen_combs))
    print('%d totally in test data over %d' % (sum(unseen_combs.values()), testsize))
    print('ratio: %.3f' % (sum(unseen_combs.values())/testsize))
