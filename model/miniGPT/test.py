import numpy as np

def test(batches, epoch, mode, args):
    epoch_loss = 0; epoch_num_tokens = 0
    numwords = args.valsize if mode =='val'  else args.tstsize
    indices = list(range(len(batches)))
    for i, idx in enumerate(indices):
        # (batchsize, t)
        surf = batches[idx]
        loss, _acc, _ = args.model(surf)
        epoch_num_tokens += surf.size(0) * (surf.size(1)-1) # exclude start token prediction
        epoch_loss       +=  loss.sum().item()
    nll = epoch_loss / numwords
    ppl = np.exp(epoch_loss / epoch_num_tokens)
    args.logger.write(f"Epoch: {epoch}/{args.epochs} |  nll: {nll:.4f} | perplexity: {ppl:.4f}\n")
    return nll, ppl