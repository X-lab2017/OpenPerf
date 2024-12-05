import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取转换后的CSV文件
data = pd.read_csv('converted_data.csv', delimiter=';')

# 提取特征列（除去id列）
features = data.iloc[:, 1:]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# 使用K-means算法进行聚类，指定聚类数为4
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# 映射数字聚类标签到字符串标签
cluster_mapping = {0: 'leader', 1:'maintainer' , 2: 'contributor', 3: 'observer'}
data['cluster'] = data['cluster'].map(cluster_mapping)

# 保存聚类结果到新的CSV文件
data.to_csv('clustered_data.csv', index=False, sep=';')

plt.scatter(data['contributions'], data['stars'], c=data['cluster'].map({'observer': 0, 'contributor': 1, 'maintainer': 2, 'leader': 3}), cmap='viridis')
plt.xlabel('Contributions')
plt.ylabel('Stars')
plt.title('K-means Clustering')
plt.colorbar(ticks=[0, 1, 2, 3], format=plt.FuncFormatter(lambda val, loc: ['observer', 'contributor', 'maintainer', 'leader'][loc]))
plt.show()
print("聚类完成并保存为 'clustered_data.csv'")