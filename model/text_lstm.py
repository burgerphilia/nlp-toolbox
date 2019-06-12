import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class TextLSTM(nn.Module):

    def __init__(self, hidden_dim, vocab_size, embedding_dim=200, n_classes=2, weights=None):
        
        super(TextLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # vetorization
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        if weights:
            
            self.embeddings.weight = nn.Parameter(weights, requires_grad=False)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, n_classes), 
                                 nn.Softmax(dim=1))

    def forward(self, sequence, lengths):
        
        embeds = self.embeddings(sequence)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True) 
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(packed)
        out = self.fc(final_hidden_state[-1])
        
        return out
    
    def get_last_hidden_neuron(self, sequence, lengths):
        
        embeds = self.embeddings(sequence)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True) 
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(packed)
        
        return final_hidden_state[-1]