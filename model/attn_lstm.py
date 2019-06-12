import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttnLSTM(nn.Module):
    
    def __init__(self, hidden_dim, vocab_size, embedding_dim, num_layers=1, dropout=0.2, n_classes=4):
        
        super(AttnLSTM, self).__init__()
        
        # Keep for reference
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=(0 if num_layers ==  1 else dropout), 
                            bidirectional=True)
        self.W_s1 = nn.Linear(2*hidden_dim, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc = nn.Sequential(nn.Linear(30*2*hidden_dim, 2000), 
                                nn.Linear(2000, 200))
        self.output_layer = nn.Linear(200, n_classes)
        
    def attention(self, lstm_out):
        
        attn_weight = self.W_s2(torch.tanh(self.W_s1(lstm_out)))
        attn_weight = attn_weight.permute(0, 2, 1)
        attn_weight = F.softmax(attn_weight, dim=2)
        
        return attn_weight
    
    def forward(self, sequence, lengths):
        
        embeds = self.embedding(sequence)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True) 
        
        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_weight = self.attention(lstm_out)
        hidden = torch.bmm(attn_weight, lstm_out)
        
        fc_out = self.fc(hidden.view(-1, hidden.size()[1]*hidden.size()[2]))
        out = self.output_layer(fc_out)
        
        return out
    
    def get_last_hidden_neuron(self, sequence, lengths):
        
        embeds = self.embedding(sequence)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True) 
        
        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_weight = self.attention(lstm_out)
        hidden = torch.bmm(attn_weight, lstm_out)
        
        fc_out = self.fc(hidden.view(-1, hidden.size()[1]*hidden.size()[2]))
        
        return fc_out