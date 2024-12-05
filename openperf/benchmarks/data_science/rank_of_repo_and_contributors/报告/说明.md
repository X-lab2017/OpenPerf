# Github影响力分析

### 数据来源

ods_github_log：

PullRequestEvent类型，时间在16年至21年之间的repo_id, repo_name, user_id, user_login, ceated_at数据

WatchEvent类型，时间在16年至21年之间，repo_id在PullRequestEvent类型取出的repo_id中的全部记录

ods_github_users：

login在PullRequestEvent类型取出的actor_login中的login, followers, lastupdatedat数据

### 代码说明

#### 运行环境

JupyterLab Version 3.2.1

#### birank_pagerank.ipynb

使用数据：直接从ods_github_log和ods_github_users获取
内容：数据预处理，birank算法实现，pagerank算法实现，watch数据与follower数据获取，Spearman相关性分析
输出文件：为横向比较服务，输出两个算法得到的前1000的仓库和开发者ID，分别存放在pagerank_repo_top1000.csv、pagerank_actor_top1000.csv、birank_repo_top1000.csv、birank_actor_top1000.csv

#### burstbirank.ipynb

使用数据：直接从ods_github_log和ods_github_users获取
内容：数据预处理，burstbirank算法实现，watch数据与follower数据获取，Spearman相关性分析
输出文件：为横向比较服务，输出算法得到的前1000的仓库和开发者ID，分别存放在burstbirank_repo_top1000.csv、burstbirank_actor_top1000.csv

#### result.ipynb

使用数据：（三个算法的排名Top50的开发者和仓库id），即：burstbirank_repo_top1000.csv、burstbirank_actor_top1000.csv、pagerank_repo_top1000.csv、pagerank_actor_top1000.csv、birank_repo_idname_top100.csv、birank_actor_idname_top100.csv
内容：对BiRank算法判定的Top50的开发者和仓库，合并它们在BurstBiRank和PageRank算法中的排名

#### evalation.ipynb

使用数据：（三个算法的排名Top1000的开发者和仓库id），即：burstbirank_repo_top1000.csv、burstbirank_actor_top1000.csv、pagerank_repo_top1000.csv、pagerank_actor_top1000.csv、birank_repo_top1000.csv、birank_actor_top1000.csv
内容：对三个算法得到的结果交叉验证，实现横向比较。分别得出开发者/仓库Top10、50、100、200、500、1000的准确度、召回值和F1值，并作图可视化