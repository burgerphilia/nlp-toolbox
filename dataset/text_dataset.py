from os import listdir
from os.path import join, isfile
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    
    def __init__(self, root, vocab):
        
        self.root = root
        self.text = np.load(root+'text.npy')
        self.label = np.load(root+'label.npy')
        self.vocab = vocab
    
    def __len__(self):
        
        return self.text.shape[0]
        
    def __getitem__(self, idx):
        
        tokens = tokenize(self.text[idx])
        indexed_tokens = [self.vocab(tok) for tok in tokens]
        
        indexed_tokens = torch.tensor(indexed_tokens, dtype=torch.long)
        label = torch.tensor(self.label[idx], dtype=torch.long)
        
        return indexed_tokens, label

def collate_fn(data):
    
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*data)
    
    # Merge sequences (from tuple of 1D tensor to 2D tensor).
    lengths = [len(seq) for seq in sequences]
    targets = torch.zeros(len(sequences), max(lengths)).long()
    for i, seq in enumerate(sequences):
        eos = lengths[i]
        targets[i, :eos] = seq[:eos]
    
    # Merge labels (from tuple of 1D tensor to 2D tensor).
    labels = torch.stack(labels, 0)
        
    return targets, labels, lengths

def get_loader(root, vocab, batch_size, shuffle, num_workers=None):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    
    dataset = TextDataset(root=root, vocab=vocab)
    
    # Data loader for CS dataset
    # This will return (sequences, labels, lengths) for each iteration.
    # sequences: a tensor of shape (batch_size, padded length).
    # lables: a tensor of shape (batch_size, ).
    # lengths: a list indicating valid length for each sequeces. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader