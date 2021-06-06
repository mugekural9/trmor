from torch import nn, optim
import torch
from vae import VAE
from dec_lstm import LSTMDecoder
from enc_lstm import LSTMEncoder
from data import readdata, get_batches
import argparse, time, sys
import numpy as np


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

def calc_au(model, test_data_batch, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for _, batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for _, batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


def calc_mi(model, test_data_batch):
    mi = 0
    num_examples = 0
    for _, batch_data in test_data_batch:
        batch_size = batch_data.size(0)
        num_examples += batch_size
        mutual_info = model.calc_mi_q(batch_data)
        mi += mutual_info * batch_size

    return mi / num_examples

def test(model, test_data_batch, mode, args, verbose=True):
    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, nsamples=args.nsamples)

        assert(not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()

    mutual_info = calc_mi(model, test_data_batch)

    test_loss = (report_rec_loss  + report_kl_loss) / report_num_sents

    nll = (report_kl_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    if verbose:
        print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
               (mode, test_loss, report_kl_loss / report_num_sents, mutual_info,
                report_rec_loss / report_num_sents, nll, ppl))
        sys.stdout.flush()

    return test_loss, nll, kl, ppl, mutual_info



parser = argparse.ArgumentParser(description='VAE mode collapse study')
args = parser.parse_args()
args.batch_size = 16
args.dec_nh = 1024
args.aggressive = True
args.epochs = 10
args.kl_start = 0.1
args.warm_up = 10
args.momentum = 0
args.ni = 512
args.enc_nh = 1024
args.nz = 32
args.dec_dropout_in=0.5
args.dec_dropout_out=0.5
args.nsamples = 500
args.save_path='models/lag/lag_aggressive1_kls0.10_warm10_0_0_783435.pt'
args.test_nepoch=5

# Read data and get batches...
data, _vocab = readdata()
vocab = _vocab.word2idx
vocab_size = len(vocab)
train_data = data[:10000] #69981      
vlddata = data[10000:11000]#120000]
tstdata = data[11000:12000]#data[120000:]  # 9718
train_data_batch, _ = get_batches(train_data, _vocab, args.batch_size) 
val_data_batch, _   = get_batches(vlddata, _vocab, args.batch_size) 
test_data_batch, _  = get_batches(tstdata, _vocab, args.batch_size) 


opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

        
log_niter = (len(train_data)//args.batch_size)//10
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
args.device = device
clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5



encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
args.enc_nh = args.dec_nh
decoder = LSTMDecoder(args, vocab, model_init, emb_init)
vae = VAE(encoder, decoder, args).to(device)

enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=1.0, momentum=args.momentum)
dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=1.0, momentum=args.momentum)
opt_dict['lr'] = 1.0


iter_ = decay_cnt = 0
best_loss = 1e4
best_kl = best_nll = best_ppl = 0
pre_mi = 0
aggressive_flag = True if args.aggressive else False
vae.train()

start = time.time()
kl_weight = args.kl_start
anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))


for epoch in range(args.epochs):
    print('\nepoch: ', epoch)
    report_kl_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0
    for i in np.random.permutation(len(train_data_batch)):
        batch_data = train_data_batch[i]
        batch_size, sent_len = batch_data.size()
        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size
        report_num_sents += batch_size
        # kl_weight = 1.0
        kl_weight = min(1.0, kl_weight + anneal_rate)
        sub_iter = 1
        batch_data_enc = batch_data
        burn_num_words = 0
        burn_pre_loss = 1e4
        burn_cur_loss = 0

        # Train inference network...
        while aggressive_flag and sub_iter < 100:
            print('sub_iter:', sub_iter)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            burn_batch_size, burn_sents_len = batch_data_enc.size()
            burn_num_words += (burn_sents_len - 1) * burn_batch_size
            loss, loss_rc, loss_kl = vae.loss(batch_data_enc, kl_weight, nsamples=args.nsamples)
            burn_cur_loss += loss.sum().item()
            loss = loss.mean(dim=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)
            enc_optimizer.step()
            id_ = np.random.random_integers(0, len(train_data_batch) - 1)
            batch_data_enc = train_data_batch[id_]
            if sub_iter % 15 == 0:
                burn_cur_loss = burn_cur_loss / burn_num_words
                if burn_pre_loss - burn_cur_loss < 0:
                    break
                burn_pre_loss = burn_cur_loss
                burn_cur_loss = burn_num_words = 0
            sub_iter += 1

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, nsamples=args.nsamples)
        loss = loss.mean(dim=-1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)
        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()

        # Train both networks... returned basic VAE
        if not aggressive_flag:
            enc_optimizer.step()

        # Train model network...
        dec_optimizer.step()
        iter_ += 1

        # End of epoch, check if mi still climbing...
        if aggressive_flag and (iter_ % len(train_data_batch)) == 0:
            vae.eval()
            cur_mi = calc_mi(vae, val_data_batch)
            vae.train()
            print("pre mi:%.4f. cur mi:%.4f" % (pre_mi, cur_mi))
            if cur_mi - pre_mi < 0:
                aggressive_flag = False
                print("STOP BURNING")
            pre_mi = cur_mi

        # Report iteration results...
        # report_rec_loss += loss_rc.item()
        # report_kl_loss += loss_kl.item()
        # if True: #iter_ % log_niter == 0:
        #     train_loss = (report_rec_loss  + report_kl_loss) / report_num_sents
        #     if aggressive_flag or epoch == 0:
        #         vae.eval()
        #         with torch.no_grad():
        #             mi = calc_mi(vae, val_data_batch)
        #             au, _ = calc_au(vae, val_data_batch)
        #         vae.train()
        #         print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
        #                'au %d, time elapsed %.2fs' %
        #                (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
        #                report_rec_loss / report_num_sents, au, time.time() - start))
        #     else:
        #         print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
        #                'time elapsed %.2fs' %
        #                (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
        #                report_rec_loss / report_num_sents, time.time() - start))
        #     sys.stdout.flush()
        #     report_rec_loss = report_kl_loss = 0
        #     report_num_words = report_num_sents = 0



    print('kl weight %.4f' % kl_weight)
    vae.eval()

    with torch.no_grad():
        loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
        au, au_var = calc_au(vae, val_data_batch)
        print("%d active units" % au)
        # print(au_var)
    if loss < best_loss:
        print('update best loss')
        best_loss = loss
        best_nll = nll
        best_kl = kl
        best_ppl = ppl
        torch.save(vae.state_dict(), args.save_path)
    if loss > opt_dict["best_loss"]:
        opt_dict["not_improved"] += 1
        if opt_dict["not_improved"] >= decay_epoch and epoch >=15:
            opt_dict["best_loss"] = loss
            opt_dict["not_improved"] = 0
            opt_dict["lr"] = opt_dict["lr"] * lr_decay
            vae.load_state_dict(torch.load(args.save_path))
            print('new lr: %f' % opt_dict["lr"])
            decay_cnt += 1
            enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
            dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
    else:
        opt_dict["not_improved"] = 0
        opt_dict["best_loss"] = loss
    if decay_cnt == max_decay:
        break
    if epoch % args.test_nepoch == 0:
        with torch.no_grad():
            loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)
    vae.train()


