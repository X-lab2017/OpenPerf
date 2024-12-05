'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        可以不使用接口的tokenizer, 只需在api_decode中对应修改为自己的tokenizer即可
        需要调用vocab的add_label方法扩容vocab
'''

import numpy as np

class Keys:
    sentence = 'sentence'
    entity_mentions = 'golden-entity-mentions'
    text = 'text'
    type = 'entity-type'
    start = 'start'
    end = 'end'
    index = 'index'

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text

def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        if instance.get(Keys.entity_mentions, None) is not None:
            for entity in instance[Keys.entity_mentions]:
                vocab.add_label(entity[Keys.type])
            entity_num += len(instance[Keys.entity_mentions])
    return entity_num

def api_encode(data, tokenizer, vocab):
        bert_inputs = []
        grid_labels = []
        grid_mask2d = []
        dist_inputs = []
        entity_text = []
        pieces2word = []
        sent_length = []

        fill_vocab(vocab, data)

        for idx, instance in enumerate(data):
            sentence = instance[Keys.sentence]
            if len(sentence) == 0 or len(sentence) >= 512:
                continue

            tokens = [tokenizer.tokenize(word) for word in sentence]
            pieces = [piece for pieces in tokens for piece in pieces]
            _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
            _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

            length = len(sentence)
            _grid_labels = np.zeros((length, length), dtype=np.int64)
            _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
            _dist_inputs = np.zeros((length, length), dtype=np.int64)
            _grid_mask2d = np.ones((length, length), dtype=bool)

            if tokenizer is not None:
                start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(start, start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    start += len(pieces)

            for k in range(length):
                _dist_inputs[k, :] += k
                _dist_inputs[:, k] -= k

            for i in range(length):
                for j in range(length):
                    if _dist_inputs[i, j] < 0:
                        _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                    else:
                        _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
            _dist_inputs[_dist_inputs == 0] = 19

            _entity_text = []
            if instance.get(Keys.entity_mentions, None) is not None:
                for entity in instance[Keys.entity_mentions]:
                    # b, e, type = entity[Keys.start], entity[Keys.end], entity[Keys.type]
                    # if b < 0: continue
                    # if e > len(sentence): e = len(sentence)
                    index, type = entity['index'], entity[Keys.type]
                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        _grid_labels[index[i], index[i + 1]] = vocab.label_to_id(vocab.SUC)
                    _grid_labels[index[-1], index[0]] = vocab.label_to_id(type)
                    _entity_text.append(convert_index_to_text(index, vocab.label_to_id(type)))
            _entity_text = set(_entity_text)

            sent_length.append(length)
            bert_inputs.append(_bert_inputs)
            grid_labels.append(_grid_labels)
            grid_mask2d.append(_grid_mask2d)
            dist_inputs.append(_dist_inputs)
            pieces2word.append(_pieces2word)
            entity_text.append(_entity_text)

        return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text