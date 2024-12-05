import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 定义一个简单的MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 读取CSV文件
data = pd.read_csv('kaiyuan/cleaned_data.csv', delimiter=',')

# 提取特征列（除去id列）
features = data.iloc[:, 1:]

# 使用Min-Max标准化对每个特征进行缩放
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# 转换为PyTorch张量
features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

# 定义MLP模型
input_dim = features.shape[1]
hidden_dim = 64
output_dim = 16
mlp = MLP(input_dim, hidden_dim, output_dim)

# 使用MLP对特征进行处理
mlp_output = mlp(features_tensor).detach().numpy()

# 使用GMM算法进行聚类，指定聚类数为4
gmm = GaussianMixture(n_components=4, random_state=42)
data['cluster'] = gmm.fit_predict(mlp_output)

# 映射数字聚类标签到字符串标签
cluster_mapping = {0: 'observer', 1: 'contributor', 2: 'maintainer', 3: 'leader'}
data['cluster'] = data['cluster'].map(cluster_mapping)

# 保存聚类结果到新的CSV文件
data.to_csv('clustered_data.csv', index=False, sep=';')




print("聚类完成并保存为 'clustered_data.csv'")