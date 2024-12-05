# task7:  find the influential developers and repositories in Github open source community 

## 1.dataset

收集的数据是GitHub开发者协作网络数据集，节点包括开发者和仓库，边表示开发者与仓库的交互协作关系和开发者的follow关系（包括pr,issue,folk,push,release等等）节点有8827个开发者，9000个仓库，边共有241200。数据收集来自REST API和clickhouse。这个协作网络作为信息传播网络

## 2.seed selection

种子节点分为两个步骤：一是对每个开发者和仓库进行影响力计算，然后对影响力排序，选择前K个节点作为种子节点，放入传播网络中传播。

其中，影响力计算的使用的openrank算法，传播网络是使用独立级联网络（IC）。

最后对比其他两个中心度算法degree和PageRank，评价指标使用的是传播规模与传播速度，从下图可以看到openrank比度中心线算法和PageRank都要好。

<image src="ic_task7.png">;

<image src="task7.png">;