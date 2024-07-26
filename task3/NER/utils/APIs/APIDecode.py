'''
decode api: 将model预测出的数据转变成理想的格式
    tips: 
        需要与Model配合, Model forward的输出即此api输入
'''

from collections import defaultdict, deque

def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text

def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

def api_decode(outputs, entities, length):

    class Node:
        def __init__(self):
            self.THW = []                # [(tail, type)]
            self.NNW = defaultdict(set)   # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()
    for instance, ent_set, l in zip(outputs, entities, length):
        predicts = []
        nodes = [Node() for _ in range(l)]
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur+1):
                # THW
                if instance[cur, pre] > 1: 
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head,cur)].add(cur)
                    # post nodes
                    for head,tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head,tail)].add(cur)
            # entity
            for tail,type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur,tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])
        
        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        ent_c += len(predicts.intersection(ent_set))
    return ent_c, ent_p, ent_r, decode_entities