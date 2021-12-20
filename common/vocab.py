import random
import torch
import numpy as np
from collections import defaultdict, Counter

class VocabEntry(object):
    """docstring for Vocab"""
    def __init__(self, word2id=None):
        super(VocabEntry, self).__init__()

        if word2id:
            self.word2id = word2id
            self.unk_id = word2id['<unk>']
        else:
            self.word2id = dict()
            self.unk_id = 3
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = self.unk_id

        self.id2word_ = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid

        else:
            return self[word]

    def id2word(self, wid):
        return self.id2word_[wid]

    def decode_sentence(self, ids):
        decoded_sentence = []
        for wid_t in ids:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence
    
    def encode_sentence(self, sentence):
        encode_sentence = []
        for char in sentence:
            encode_sentence.append(self.word2id[char])
        return encode_sentence

    def decode_sentence_2(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            decoded_sentence.append(self.id2word_[wid_t])
        return decoded_sentence


    @staticmethod
    def from_corpus(fname):
        vocab = VocabEntry()
        with open(fname) as fin:
            for line in fin:
                _ = [vocab.add(word) for word in line.split()]

        return vocab