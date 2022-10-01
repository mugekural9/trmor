#!/bin/bash
#SBATCH --job-name=early-ml # Job name
#SBATCH --nodes=1 # Run on a single node
#SBATCH --ntasks-per-node=5 # Run a single task on each node
#SBATCH --partition=ai # Run in ai queue
#SBATCH --qos=ai # Run in qos (ai)
#SBATCH --account=ai # Run account (ai)
#SBATCH --time=96:00:00 # Time limit days-hours:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --constrain=tesla_t4
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mugekural@ku.edu.tr # Where to send mail

echo "running"
DIR=$(pwd) && export PYTHONPATH="$DIR"

# early-sup
python /kuacc/users/mugekural/workfolder/dev/git/trmor/model/vqvae/sig2016/early_sup/vqvae_train_kl_bi_early_sup_gradual_nobias_lstm.py \
--kl_max 0.1 --lang maltese 