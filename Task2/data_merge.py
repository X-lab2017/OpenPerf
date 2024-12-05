import pandas as pd
import os

# Define the directory containing the CSV files
directory = 'C:/Users/Silly/Desktop/data/'

# List of CSV filenames
csv_files = [
    'anhuidianxinzhidao_filter.csv',
    'baoxianzhidao_filter.csv',
    'financezhidao_filter.csv',
    'lawzhidao_filter.csv',
    'liantongzhidao_filter.csv',
    'nonghangzhidao_filter.csv'
]

# Initialize an empty list to hold dataframes
dfs = []

# Read each CSV file and append the dataframe to the list
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(directory, csv_file))
    dfs.append(df)

# Concatenate all dataframes in the list
merged_df = pd.concat(dfs, ignore_index=True)

# Output the final merged dataframe to a new CSV file
output_file = os.path.join(directory, 'merged_output.csv')
merged_df.to_csv(output_file, index=False)

print(f'Merged CSV file saved to {output_file}')
