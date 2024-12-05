import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 读取 CSV 数据
data = pd.read_csv('kaiyuan/cleaned_data.csv')

# 提取特征和标签
X = data.drop(['user'], axis=1).values  # 假设 'user' 列是用户 ID

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 定义 MLP 模型
input_dim = X_tensor.shape[1]
hidden_dim = 100  # 可以根据需要调整
output_dim = 2  # MLP 输出的特征维度，可以根据需要调整
mlp = MLP(input_dim, hidden_dim, output_dim)

# 使用 DataLoader 进行批处理
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 提取 MLP 特征
mlp.eval()
with torch.no_grad():
    mlp_features = []
    for batch in dataloader:
        features = mlp(batch[0])
        mlp_features.append(features)
    mlp_features = torch.cat(mlp_features).numpy()

# 使用 GMM 进行聚类
gmm = GaussianMixture(n_components=4, random_state=0)
clusters = gmm.fit_predict(mlp_features)

# 计算轮廓系数
silhouette_avg = silhouette_score(mlp_features, clusters)
print(f"Average Silhouette Score: {silhouette_avg}")