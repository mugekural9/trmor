import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random, time
from test import test
from common.utils import *

def train(data, logger, args):
    trnbatches, valbatches, tstbatches = data
    #scheduler = ReduceLROnPlateau(args.opt, 'min', verbose=1, factor=0.5)
    numbatches = len(trnbatches)
    indices = list(range(numbatches))
    random.seed(0)
    best_loss = 1e4
    start_time = time.time()
    
    trn_loss_values, trn_acc_values= [], []
    val_loss_values, val_acc_values= [], []
    for epoch in range(args.epochs):
        epoch_loss = 0; epoch_acc = 0; epoch_error = 0; epoch_num_tokens = 0
        epoch_wrong_predictions = []
        epoch_correct_predictions = []
        random.shuffle(indices) # this breaks continuity if there is
        for i, idx in enumerate(indices):
            args.model.zero_grad()
            # (batchsize, t)
            surf = trnbatches[idx]

            loss, _acc, _ = args.model(surf)
            batch_loss = loss.sum() #mean(dim=-1)
            batch_loss.backward()
            args.opt.step()
            correct_tokens, num_tokens, wrong_tokens, wrong_predictions, correct_predictions = _acc
            epoch_num_tokens += num_tokens
            epoch_loss       += batch_loss.item()
            epoch_acc        += correct_tokens
            epoch_error      += wrong_tokens
            epoch_wrong_predictions += wrong_predictions
            epoch_correct_predictions += correct_predictions

        nll_train = epoch_loss / numbatches
        ppl_train = np.exp(epoch_loss / epoch_num_tokens)
        acc_train = epoch_acc / epoch_num_tokens

        trn_loss_values.append(nll_train)
        trn_acc_values.append(acc_train)
        logger.info(f"Epoch: {epoch}/{args.epochs} | avg_train_loss: {nll_train:.4f} | perplexity: {ppl_train:.4f} | train_accuracy: {acc_train:.4f}")

        # File Operations
        f1 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_wrong_predictions.txt", "w")
        f2 = open(args.results_file_name + "/"+str(args.epochs)+"epochs_trn_correct_predictions.txt", "w")
        for i in epoch_wrong_predictions:
            f1.write(i+'\n')
        for i in epoch_correct_predictions:
            f2.write(i+'\n')
        f1.close(); f2.close()
        
        # Validation
        args.model.eval()
        with torch.no_grad():
            nll_test, ppl_test, acc_test = test(valbatches, epoch, "val", logger, args)
            loss = nll_test
        val_loss_values.append(nll_test)
        val_acc_values.append(acc_test)
        #scheduler.step(nll)

        if loss < best_loss:
            logger.info('Update best val loss\n')
            best_loss = loss
            best_ppl = ppl_test
            torch.save(args.model.state_dict(), args.save_path)
            
        #logger.info("\n")
        logging.info("\n")
        args.model.train()

    end_time = time.time()
    training_time = convert(abs(end_time - start_time))
    logger.info(f"\n\n---Final Results---")
    logger.info(f"Epochs: {args.epochs}, Batch Size: {args.batchsize}, lr: {args.lr}, train_loss: {nll_train:.4f}, val_loss: {nll_test:.4f}")
    logger.info(f"Training Time: {training_time}\n")
    plot_curves(args.task, args.mname, args.fig, args.axs, trn_loss_values, val_loss_values, args.plt_style, 'loss')