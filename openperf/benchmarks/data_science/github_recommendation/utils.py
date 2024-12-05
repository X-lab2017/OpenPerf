import numpy as np

import torch
import torch.nn.functional as F
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


activation_getter = {'iden': lambda x: x, 'relu': F.relu, 'tanh': torch.tanh, 'sigm': torch.sigmoid}


def str2bool(v):
    return v.lower() in ('true')


class CustomDataset(Dataset):
    # TODO: add negative sampling
    def __init__(self, L, T, sequences, languages, descriptions_parsed, train=False, neg_samples=3):
        self.L, self.T = L, T
        
        self.input_items = sequences.sequences[:, :L]
        self.target_items = sequences.sequences[:, L:]
        self.num_seq = self.input_items.shape[0]
        self.user_ids = sequences.user_ids - 1 # num_seq
        self.num_items = sequences.num_items
        
        self.languages = languages # dict, num_item
        self.descriptions = descriptions_parsed # dict, num_item
        
        self.train = train
        self.neg_samples = 3*T
        
        if self.train:
            # negative samples
            self.matrix = dok_matrix((np.max(self.user_ids)+1, self.num_items), dtype=np.bool)
            for user, seq in zip(self.user_ids, sequences.sequences):
                self.matrix[user-1, seq] = True

            self.matrix = self.matrix.tocsr()
        
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        input_items = self.input_items[index]
        input_items_lang = torch.LongTensor([self.languages[i] for i in input_items])
        input_items_desc = [self.descriptions[i] for i in input_items]
        
        target_items = self.target_items[index]
        target_items_lang = torch.LongTensor([self.languages[i] for i in target_items])
        target_items_desc = [self.descriptions[i] for i in target_items]
        
        # for negative sampling
        if self.train:
            unwatched_items = np.setdiff1d(np.arange(self.num_items), self.matrix[user_id].nonzero()[1])
            neg_items = np.random.choice(unwatched_items, size=self.neg_samples, replace=False)
            
            neg_items_lang = torch.LongTensor([self.languages[i] for i in neg_items])
            neg_items_desc = [self.descriptions[i] for i in neg_items]
            
            neg_samples = (torch.LongTensor(neg_items), neg_items_lang, neg_items_desc)
        else:
            neg_samples = None
        
        return (torch.LongTensor([user_id]),
                torch.LongTensor(input_items), input_items_lang, input_items_desc,
                torch.LongTensor(target_items), target_items_lang, target_items_desc,
                neg_samples)

    def __len__(self):
        return self.num_seq
