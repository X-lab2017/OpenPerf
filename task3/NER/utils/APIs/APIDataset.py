'''
Dataset api: 与api_encode配合, 将api_encode的返回结果构造成Dataset方便Pytorch调用
    tips:
        注意如果数据长度不一需要编写collate_fn函数, 若无则将collate_fn设为None
'''

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class APIDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item]

    def __len__(self):
        return len(self.bert_inputs)

    # collate_fn = None

    def collate_fn(self, data):
        bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list, zip(*data))

        max_tok = np.max(sent_length)
        sent_length = torch.LongTensor(sent_length)
        max_pie = np.max([x.shape[0] for x in bert_inputs])
        bert_inputs = pad_sequence(bert_inputs, True)
        batch_size = bert_inputs.size(0)

        def fill(data, new_data):
            for j, x in enumerate(data):
                new_data[j, :x.shape[0], :x.shape[1]] = x
            return new_data

        dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
        dist_inputs = fill(dist_inputs, dis_mat)
        labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
        grid_labels = fill(grid_labels, labels_mat)
        mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
        grid_mask2d = fill(grid_mask2d, mask2d_mat)
        sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
        pieces2word = fill(pieces2word, sub_mat)

        return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text