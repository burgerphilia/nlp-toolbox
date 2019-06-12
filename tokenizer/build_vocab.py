import numpy as np
import nltk
import pickle
import argparse
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        
        self.token_to_idx = dict()
        self.idx_to_token = dict()
        self.idx = 0

    def add_token(self, token):
        # to do: replace try, catch stmt
        if not token in self.token_to_idx:

            self.token_to_idx[token] = self.idx
            self.idx_to_token[self.idx] = token
            self.idx += 1

    def __call__(self, token):

        if not token in self.token_to_idx:
            
            return self.token_to_idx['<UNK>']

        return self.token_to_idx[token]

    def __len__(self):

        return len(self.token_to_idx)

def tokenize(seq):
    
    token = list()
    
    for s in seq:
        
        tmp = s.replace(" ", "")
        token.extend(list(tmp))
        
        
    return token

def build_vocab(root, threshold=5, tokenizer=None):
    """to do: replace data path to json """
    sequences = np.load(root+'text.npy')
    counter = Counter()
    
    #tokenize - add nltk, konlpy,mecab, WPM
    for i, seq in enumerate(sequences):
        
        tokens = tokenize(seq)
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i+1, sequences.shape[0]))

    # If the word frequency is less than 'threshold', then the word is discarded.
    tokens = [token for token, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_token('<PAD>')
    vocab.add_token('<START>')
    vocab.add_token('<END>')
    vocab.add_token('<UNK>')

    # Add the words to the vocabulary.
    for i, tok in enumerate(tokens):
        
        vocab.add_token(tok)
    
    return vocab