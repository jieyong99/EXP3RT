import json
import os

def merge_json_files(input_prefix, output_file):
    merged_data = []
    
    
    files = [f for f in os.listdir('data/amazon-book/rating_bias') if f.startswith(input_prefix) and f.endswith('.json')]
    
    
    files.sort()
    
    for file in files:
        file_path = os.path.join('data/amazon-book/rating_bias', file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            merged_data.extend(data)
    
    
    output_path = os.path.join('data/amazon-book/rating_bias', output_file)
    with open(output_path, 'w') as f:
        json.dump(merged_data, f)
    
    print(f"Merged {len(files)} files into {output_file}")
    print(f"Total items in merged file: {len(merged_data)}")


input_prefix = 'train_'
output_file = 'train.json'

merge_json_files(input_prefix, output_file)