import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 读取 CSV 文件
data = pd.read_csv('data.csv',encoding_errors='ignore')

# 选择有用的特征
features = ['login', 'id', 'name', 'company', 'email', 'bio', 'followers', 'following', 'stars', 'commits']

# 准备输入和输出变量
X = data[features]
y = data['location']

# 对文本特征进行编码
label_encoders = {}
for column in ['login', 'name', 'company', 'email', 'bio']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 预测测试集的 location
predictions = model.predict(X_test)

# 评估模型性能（例如，使用准确率）
print(accuracy_score(y_test, predictions))

