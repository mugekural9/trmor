import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random, time
from test import test
from common.utils import *

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    #scheduler = ReduceLROnPlateau(args.opt, 'min', verbose=1, factor=0.5)
    indices = list(range(len(trnbatches)))
    #random.seed(0)
    best_loss = 1e4
    start_time = time.time()
    numwords = args.trnsize

    trn_loss_values, trn_acc_values= [], []
    val_loss_values, val_acc_values= [], []
    for epoch in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = []; epoch_correct_predictions = []
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx]
            loss, _acc, _ = args.model(surf)
            batch_loss = loss.mean()
            batch_loss.backward()
            args.opt.step()
            epoch_num_tokens += surf.size(0) * (surf.size(1)-1) # exclude start token prediction
            epoch_loss       += loss.sum().item()

        nll_train = epoch_loss / numwords
        ppl_train = np.exp(epoch_loss / epoch_num_tokens)
        trn_loss_values.append(nll_train)
        args.logger.write(f"Epoch: {epoch}/{args.epochs} | nll: {nll_train:.4f} | perplexity: {ppl_train:.4f}\n")

        # Validation
        args.model.eval()
        with torch.no_grad():
            nll_test, ppl_test = test(valbatches, epoch, "val", args)
            loss = nll_test
        val_loss_values.append(nll_test)
        #scheduler.step(nll_test)

        if loss < best_loss:
            args.logger.write('Update best val loss\n')
            best_loss = loss
            best_ppl = ppl_test
            torch.save(args.model.state_dict(), args.save_path)
            
        args.logger.write("\n")
        args.model.train()

    #end_time = time.time()
    #training_time = abs(end_time - start_time)
    #args.logger.write(f"\n\n---Final Results---")
    #args.logger.write(f"Epochs: {args.epochs}, Batch Size: {args.batchsize}, lr: {args.lr}, train_loss: {nll_train:.4f}, val_loss: {nll_test:.4f}")
    #args.logger.write(f"Training Time: {training_time}\n")
    plot_curves(args.task, args.mname, args.fig, args.axs, trn_loss_values, val_loss_values, args.plt_style, 'loss')