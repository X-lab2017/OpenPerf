'''
data process: 数据处理, 包括 标签Vocab 和 数据处理类
    tips:
        其中标签Vocab实例化对象必须在api_encode中被调用(add_label)
'''

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from APIs.APIDataset import APIDataset
from APIs.APIEncode import api_encode
from APIs.APIDecode import api_decode

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LabelVocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'   # 表连续随后的

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.label2id)

    def label_to_id(self, label):
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


class Processor:
    def __init__(self, config) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir=config.cache_path)
        self.vocab = LabelVocabulary()

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        return api_encode(data, self.tokenizer, self.vocab)
    
    def decode(self, outputs, entities, length):
        return api_decode(outputs, entities, length)

    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        self.config.label_num = len(self.vocab.label2id)
        self.config.vocab = self.vocab
        return APIDataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn, num_workers=2)

