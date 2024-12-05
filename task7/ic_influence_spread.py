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
    # print(active_nodes_count)

    return active_nodes_count[-1]


type=['pr','issue','folk','push','release']
adj={}
dev_num=8827
repo_num=9000
for i in type:
    adj[i]=[[0]*repo_num for _ in range(dev_num)]

repo_file=r'D:\06 dataset\openrank\repo9000_dev8827\repoid_openrank.csv'
dev_file=r'D:\06 dataset\openrank\repo9000_dev8827\actorid_openrank.csv'
repo=pd.read_csv(repo_file,header=None).iloc[:,0]
dev=pd.read_csv(dev_file,header=None).iloc[:,0]
repo_mapping={str(repo[i]):i for i in range(len(repo))}
repo_rev_mapping={i:str(repo[i]) for i in range(len(repo))}
dev_mapping={str(dev[i]):i for i in range(len(dev))}
dec_rec_mapping={i:str(dev[i]) for i in range(len(dev))}
path=r'D:\06 dataset\openrank\repo9000_dev8827\actorid_repoid_type.csv'
with open(path ,'r')as f:
    relation=csv.reader(f)
    for row in relation:
        dev=dev_mapping[row[0]]
        repo=repo_mapping[row[1]]
        if row[2] in('PullRequestEvent', 'PullRequestReviewEvent', 'PullRequestReviewCommentEvent'):
            adj['pr'][dev][repo]+=1
        elif row[2] in('IssuesReactionEvent', 'IssueCommentEvent', 'CommitCommentEvent'):
            adj['issue'][dev][repo]+=1
        elif row[2]=='ForkEvent':
            adj['folk'][dev][repo]+=1
        elif row[2]=='PushEvent':
            adj['push'][dev][repo]+=1
        elif row[2]=='ReleaseEvent':
            adj['release'][dev][repo]+=1
# print(adj)
def safe_divide(numerator, denominator):
    return np.where(denominator == 0, 0, numerator / denominator)

ar1 = safe_divide(np.array(adj['pr']), np.array(adj['pr']).sum(axis=1, keepdims=True))
ar2 = safe_divide(np.array(adj['issue']), np.array(adj['issue']).sum(axis=1, keepdims=True))
ar3 = safe_divide(np.array(adj['push']), np.array(adj['push']).sum(axis=1, keepdims=True))
ar4 = safe_divide(np.array(adj['release']), np.array(adj['release']).sum(axis=1, keepdims=True))
ar5 = safe_divide(np.array(adj['folk']), np.array(adj['folk']).sum(axis=1, keepdims=True))

total_adj=0.3*ar1+0.2*ar2+0.25*ar3+0.15*ar4+0.1*ar5
row_sums = total_adj.sum(axis=1, keepdims=True)
normalized_matrix = safe_divide(total_adj , row_sums)
#print(normalized_matrix)
pr_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\pr.csv',header=None)
issue_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\issue.csv',header=None)
push_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\push.csv',header=None)
release_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\release.csv',header=None)
folk_df = pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\folk.csv',header=None)
repo_node=pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\repoid_openrank.csv',header=None)
actor_node=pd.read_csv(r'D:\06 dataset\openrank\repo9000_dev8827\actorid_openrank.csv',header=None)

G = nx.Graph()
# 添加开发者和仓库节点
G.add_nodes_from(actor_node.iloc[:,0], bipartite=0, node_type='developer')
G.add_nodes_from(repo_node.iloc[:,0], bipartite=1, node_type='repository')

G.add_edges_from([(row[0], row[1]) for index, row in pr_df.iterrows()])
G.add_edges_from([(row[0], row[1]) for index, row in issue_df.iterrows()])
G.add_edges_from([(row[0], row[1]) for index, row in push_df.iterrows()])
G.add_edges_from([(row[0], row[1]) for index, row in release_df.iterrows()])
G.add_edges_from([(row[0], row[1]) for index, row in folk_df.iterrows()])
for u,v in G.edges():
    G[u][v]['weight']=normalized_matrix[dev_mapping[str(u)]][repo_mapping[str(v)]]

node_degrees = dict(G.degree())
sorted_nodes_by_degree = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
sorted_nodes = [node for node, degree in sorted_nodes_by_degree]

openrank=[]
pagerank=[]
degree=[]
max_iter_num = 30  
top_actor_df1 = pd.read_csv(r'top_developers.csv',header=None)
top_repo_def = pd.read_csv(r'top_repositories.csv',header=None)
for i in range(1,1001,50):
    openrank_seed=repo_node.iloc[:i,0].tolist()+actor_node.iloc[:i,0].tolist()
    #print(openrank_seed)
    pagerank_seed=top_actor_df1.iloc[:i,0].tolist()+top_repo_def.iloc[:i,0].tolist()
    degree_seed=sorted_nodes[:i]
    #print(pagerank_seed)
    influence_openrank = influence_maximization_ic(G, openrank_seed, max_iter_num)
    influence_pagerank = influence_maximization_ic(G, pagerank_seed, max_iter_num)
    degree_result= influence_maximization_ic(G,degree_seed,max_iter_num)
    openrank.append(influence_openrank)
    pagerank.append(influence_pagerank)
    degree.append(degree_result)
print(openrank)
print(pagerank)
plt.plot(range(1,1001,50), openrank, label='openrank',marker='o')
plt.plot(range(1,1001,50), pagerank, label='pagerank',marker='o')
plt.plot(range(1,1001,50), degree, label='degree',marker='o')
plt.xlabel('Iteration')
plt.ylabel('Number of Active Nodes')
plt.title('Influence Spread')
plt.legend(['OpenRank','PagerRank','degree'])
plt.savefig('photo/test2.png')
plt.show()
