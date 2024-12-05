import pandas as pd
import os

# Define the directory containing the CSV file
directory = 'C:/Users/Silly/Desktop/data/'

# Path to the merged CSV file
csv_file = os.path.join(directory, 'merged_output.csv')

# Read the merged CSV file
df = pd.read_csv(csv_file)

# Convert the dataframe to JSON format
json_output = df.to_json(orient='records', force_ascii=False)

# Path to the output JSON file
json_file = os.path.join(directory, 'merged_output.json')

# Write the JSON data to a file
with open(json_file, 'w', encoding='utf-8') as file:
    file.write(json_output)

print(f'JSON file saved to {json_file}')
