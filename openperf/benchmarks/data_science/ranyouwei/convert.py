import pandas as pd

def convert_k_to_number(value):
    if 'k' in value:
        return int(float(value.replace('k', '')) * 1000)
    return int(value.replace(',', ''))

def clean_numeric_column(value):
    if pd.isna(value):
        return 0
    value_str = str(value).replace(',', '').replace('"', '')
    try:
        return int(float(value_str))
    except ValueError:
        return 0
# 读取CSV文件
data = pd.read_csv('kaiyuan/data1.csv')


# 定义需要转换的列及其对应的函数
conversion_functions = {
    'star': convert_k_to_number,
    'fork': convert_k_to_number,
    'watch': convert_k_to_number,
    'issue': clean_numeric_column,
    'pull_requests': clean_numeric_column,
    'projects': clean_numeric_column,
    'commits': clean_numeric_column,
    'branches': clean_numeric_column,
    'contributers': clean_numeric_column,
    'releases':clean_numeric_column
}

# 应用转换函数
for column, func in conversion_functions.items():
    if column in data.columns:
        data[column] = data[column].apply(func)
# 处理license列，进行one-hot编码
license_dummies = pd.get_dummies(data['License'], prefix='license').astype(int)
data = pd.concat([data, license_dummies], axis=1)

# 删除原始的license列
data.drop('License', axis=1, inplace=True)

# 保存清洗后的数据到新的CSV文件
data.to_csv('kaiyuan/cleaned_data.csv', index=False)

print("数据清洗完成并保存为 'cleaned_data.csv'")