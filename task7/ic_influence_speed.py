import pandas as pd
import numpy as np
import networkx as nx
import csv
import random
import matplotlib.pyplot as plt
random.seed(42)
def influence_maximization_ic(G, seed, max_iter_num):
    # 初始化节点状态
    for node in G.nodes:
        G.nodes[node]['state'] = 0  # 0表示未激活，1表示激活

    # 设置初始种子节点
    for s in seed:
        G.nodes[s]['state'] = 1
              
    all_active_nodes,cur_new_active = seed[:],seed[:]
    active_nodes_count = [len(seed)]  # 记录每次迭代后激活节点数量的列表

    for i in range(max_iter_num):
        new_active=[]

        for v in cur_new_active:
            for nbr in G.neighbors(v):
                if G.nodes[nbr]['state'] == 0:  # 如果邻居节点未激活
                    edge_data = G.get_edge_data(v, nbr)
                    if random.uniform(0, 1) < edge_data['weight']:  # 根据边的权重激活节点
                        G.nodes[nbr]['state'] = 1
                        new_active.append(nbr)
        cur_new_active=new_active

        all_active_nodes.extend(new_active)
        active_nodes_count.append(len(all_active_nodes))
    print(active_nodes_count)

    return active_nodes_count



pr_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\pr.csv',header=None)
issue_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\issue.csv',header=None)
push_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\push.csv',header=None)
release_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\release.csv',header=None)
folk_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\folk.csv',header=None)
repo_node=pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\repoid_reponame.csv',header=None)
actor_node=pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\actorid_openrank.csv',header=None)
following=pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\follow.csv',header=None)

G = nx.Graph()
# 添加开发者和仓库节点
G.add_nodes_from(actor_node.iloc[:,0], bipartite=0, node_type='developer')
G.add_nodes_from(repo_node.iloc[:,0], bipartite=1, node_type='repository')

G.add_edges_from([(row[0], row[1]) for index, row in pr_df.iterrows()],weight=random.uniform(0, 1))
G.add_edges_from([(row[0], row[1]) for index, row in issue_df.iterrows()],weight=random.uniform(0, 1))
G.add_edges_from([(row[0], row[1]) for index, row in push_df.iterrows()],weight=random.uniform(0, 1))
G.add_edges_from([(row[0], row[1]) for index, row in release_df.iterrows()],weight=random.uniform(0, 1))
G.add_edges_from([(row[0], row[1]) for index, row in folk_df.iterrows()],weight=random.uniform(0, 1))
G.add_edges_from([(row[0], row[1]) for index, row in following.iterrows()],weight=random.uniform(0, 1))

seed1=repo_node.iloc[:2000,0].tolist()+actor_node.iloc[:2000,0].tolist()
print(len(seed1))

top_actor_df1 = pd.read_csv(r'top_developers.csv',header=None)
top_repo_def = pd.read_csv(r'top_repositories.csv',header=None)
pagerank_seed=top_actor_df1.iloc[:2000,0].tolist()+top_repo_def.iloc[:2000,0].tolist()
print(len(pagerank_seed))
max_iter_num = 30  # 迭代次数
# 在数据集上运行IC模型
node_degrees = dict(G.degree())
# 根据度对节点进行排序，从大到小
sorted_nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)

# 提取排序后的节点列表
sorted_nodes = [node for node, degree in sorted_nodes_by_degree]
degree_result= influence_maximization_ic(G,sorted_nodes[:4000],max_iter_num)
influence_result = influence_maximization_ic(G, seed1, max_iter_num)
influence_result1 = influence_maximization_ic(G, pagerank_seed, max_iter_num)


plt.plot(range(max_iter_num + 1), influence_result, label='openrank',marker='o')
plt.plot(range(max_iter_num + 1), influence_result1, label='pagerank',marker='o')
plt.plot(range(max_iter_num + 1), degree_result, label='degree',marker='o')
plt.xlabel('Iteration')
plt.ylabel('Number of Active Nodes')
plt.title('spread speed')
plt.legend(['OpenRank','PagerRank','degree'])
plt.savefig('photo/task7.png')
plt.show()
