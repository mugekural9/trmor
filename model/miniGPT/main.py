import argparse, torch, json, matplotlib, logging, os
from torch import optim
import matplotlib.pyplot as plt
from gpt3 import GPT3
from train import train
from common.utils import *
from model.miniGPT.data.data import build_data
matplotlib.use('Agg')


#### DON'T FORGET TO CHANGE THIS !!! ####
#logger_file_name = 'experiment14'              # Add ExpNUMBER !!!         
#logger_folder_name = 'EXPERIMENTS/exp14'       # Add ExpNUMBER !!!
#########################################
# Loggers
#if not os.path.exists(logger_folder_name):
#    os.mkdir(logger_folder_name)
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s | | %(levelname)s | | %(message)s')
#logger_file_name = os.path.join(logger_folder_name, logger_file_name)
#file_handler = logging.FileHandler(logger_file_name,'w')
#file_handler.setFormatter(formatter)
#logger.addHandler(file_handler)
#logger.info('Code started \n')

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

# training info
args.batchsize = 128
args.epochs = 100
args.opt= 'WAdam'
args.lr = 0.001
args.weight_decay = 0.01
args.task = 'lm'
args.seq_to_no_pad = 'surface'
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data
#args.trndata = 'data/unlabelled/top50k.wordlist.tur'
args.trndata = 'data/unlabelled/filtered_traindev.tur'
args.valdata = 'data/unlabelled/theval.tur'

args.tstdata = args.valdata
args.surface_vocab_file = args.trndata
args.maxtrnsize = 700000; args.maxvalsize = 100000; args.maxtstsize = 100000
rawdata, batches, vocab = build_data(args)
trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)

# model
args.mname = 'miniGPT'
num_layers=3
embed_dim=256
num_heads=16
block_size=128
embedding_dropout_rate=0.1 
attention_dropout_rate=0.1
residual_dropout_rate=0.1
expand_ratio = 4
args.model = GPT3(vocab=vocab,
                  num_layers=num_layers,
                  embed_dim=embed_dim,
                  num_heads=num_heads,
                  block_size=block_size,
                  embedding_dropout_rate=embedding_dropout_rate,
                  attention_dropout_rate=attention_dropout_rate,
                  residual_dropout_rate=residual_dropout_rate,
                  expand_ratio=expand_ratio)
args.model.to(args.device)

args.opt = optim.AdamW(args.model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)


# logging Results
args.modelname = 'model/'+args.mname+'/results/training/'+str(len(trndata))+'_instances/filter_TEST'
try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write("\nUsing device: {}".format(str(args.device)))
args.logger.write("\nTraining on '{}' is starting".format(args.seq_to_no_pad))
args.logger.write("\nWe now give the hyperparameters")
args.logger.write("\nNumber of Epochs: {}".format(args.epochs))
args.logger.write("\nBatch Size: {}".format(args.batchsize))
args.logger.write("\nLearning rate: {}".format(args.lr))
args.logger.write("\nWeight Decay: {}".format(args.weight_decay))
args.logger.write(f"\n==> Number of parameters {len(torch.nn.utils.parameters_to_vector(args.model.parameters()))}")
args.logger.write(f"\nNumber of Decoder Layers: {num_layers}")
args.logger.write(f"\nEmbedding Dimension: {embed_dim}")
args.logger.write(f"\nNumber of heads in Attention: {num_heads}")
args.logger.write(f"\nBlock size (spatial extent of the model for its context): {block_size}")
args.logger.write(f"\nEmbedding Dropout rate: {embedding_dropout_rate}")
args.logger.write(f"\nAttention Dropout rate: {attention_dropout_rate}")
args.logger.write(f"\nResidual Dropout rate: {residual_dropout_rate}")
args.logger.write(f"\nExpand ratio rate: {expand_ratio}")
args.logger.write('\n')



# plotting
args.fig, args.axs = plt.subplots(1)
args.plt_style = pstyle = '-'

# run
train(batches, args)
plt.savefig(args.fig_path)
