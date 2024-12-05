# coding: utf-8

from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, L, T, train, neg_samples=3):
        self.L, self.T = L, T
        self.num_users, self.num_items = train.num_users, train.num_items
        self.input_items = train.sequences.sequences
        self.target_items = train.sequences.targets
        self.num_seq = self.input_items.shape[0]
        self.user_ids = train.sequences.user_ids

        self.neg_samples = 3 * T
        self.matrix = train.tocsr().astype(np.bool)
        

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        input_items = self.input_items[index]
        input_items = torch.LongTensor(input_items)
    
        target_items = self.target_items[index]
        target_items = torch.LongTensor(target_items)
        
        unwatched_items = (~(self.matrix[user_id].toarray()[0])).nonzero()[0]
        neg_items = np.random.choice(unwatched_items, size=self.neg_samples, replace=False)
        neg_items = torch.LongTensor(neg_items)
        return torch.LongTensor([user_id]), input_items, target_items, neg_items

    def __len__(self):
        return self.num_seq


class TestDataset(Dataset):
    def __init__(self, L, T, train, test):
        self.L, self.T = L, T
        
        self.train_matrix = train.tocsr().astype(np.bool)
        
        self.test_sequences = []
        test = test.tocsr().astype(np.bool)
        _input_items = train.test_sequences.sequences
        _input_users = train.test_sequences.user_ids
        for user_id, input_items in zip(_input_users, _input_items):
            row = test[user_id].indices
            if len(row) > 0:
                self.test_sequences.append((user_id, input_items, row))
                
        self.test_num = len(self.test_sequences)
        
    def __getitem__(self, index):
        user_id, input_items, target_items = self.test_sequences[index]
        input_items = torch.LongTensor(input_items)
        pos_items = self.train_matrix[user_id].indices
        return torch.LongTensor([user_id]), input_items, target_items, pos_items
    
    def __len__(self):
        return self.test_num


def test_collate(batch):
    user_id, input_items, target_items, pos_items = zip(*batch)
    user_id = torch.stack(user_id, 0)
    input_items = torch.stack(input_items, 0)
    return user_id, input_items, target_items, pos_items
