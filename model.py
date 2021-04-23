import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input):
        T,B = input.size()
        embedded = self.embedding(input).view(T, B, -1)
        output = embedded
        output, (h0, c0) = self.lstm(output)
        return output, (h0, c0) # T,B,H, and 1,B,H


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, h0, c0):
        T, B = input.size()
        output = self.embedding(input).view(T, B, -1) # T,B,H
        output = F.relu(output)
        output, (h,c) = self.lstm(output, (h0, c0)) 
        output = self.softmax(self.out(output)) # T,B,vocabsize
        preds = output.argmax(dim=2) # T,B
        return output, (h,c), preds

    
def loss(input_tensor, target_tensor, encoder, decoder, criterion, vocab=None, is_report=False):
    T,B = input_tensor.size()
    target_inputs  =  target_tensor[:-1,:]
    target_outputs =  target_tensor[1:,:]
    input_length = input_tensor.size(0)
    target_length = target_outputs.size(0)
    _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
    decoder_output, (h,c), preds = decoder(target_inputs, encoder_hidden, encoder_cell) # preds: T,B
    acc = ((preds == target_outputs) * (target_outputs != 0)).sum() / (target_outputs != 0).sum()
    if is_report:
        tokens = []
        for t in preds[:,0]:
            token = vocab.idx2word[t]
            if token == '<eos>':
                break
            else:
                tokens.append(token)
        #print(tokens)
    loss = criterion(decoder_output.view(-1, decoder_output.size(2)), target_outputs.view(-1))
    return loss / (target_length-1), acc.item()


def test(input_tensor, target_tensor, encoder, decoder, criterion, vocab):
    T,B = input_tensor.size()
    target_outputs =  target_tensor[1:,:]
    input_length = input_tensor.size(0)
    target_length = target_outputs.size(0)
    _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
    h, c = encoder_hidden, encoder_cell
    SOS_TOKEN= vocab.word2idx['<go>']
    EOS_TOKEN= vocab.word2idx['<eos>']
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    loss = 0
    preds = []
    for di in range(target_length):
        decoder_output, (h,c), decoder_input = decoder(decoder_input, h, c)
        preds.append(decoder_input.item())
        loss += criterion(decoder_output.view(-1, decoder_output.size(2)), target_outputs[di])
    preds = torch.tensor(preds).to(device)
    acc = (preds == target_outputs.flatten()).sum() /  (target_outputs != 0).sum()
    return loss / (target_length-1), acc.item()
    

def pred(input_tensor, encoder, decoder, vocab):
    _, (encoder_hidden, encoder_cell) = encoder(input_tensor)
    h, c = encoder_hidden, encoder_cell
    SOS_TOKEN = vocab.word2idx['<go>']
    EOS_TOKEN = vocab.word2idx['<eos>']
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)
    preds = []
    for di in range(50):
        decoder_output, (h,c), decoder_input = decoder(decoder_input, h, c)
        preds.append(decoder_input.item())
        if decoder_input.item() == EOS_TOKEN:
            break
    tokens = []
    for t in preds:
        token = vocab.idx2word[t]
        if token == '<eos>':
            break
        else:
            tokens.append(token)
    return tokens
