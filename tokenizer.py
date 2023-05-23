import csv

def get_unique_column_values(csv_file, column_index):
    unique_values = set()  # Set to store unique values

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row if present

        for row in reader:
            if len(row) > column_index:  # Check if column index exists in the row
                value = row[column_index]
                unique_values.add(str(value))

    return list(unique_values)

# Usage example
csv_file_path1 = 'val_ok.csv'
csv_file_path2 = 'train_ok.csv'
column_index = 3  # Index of the column (0-based) you want to extract unique values from

list1 = get_unique_column_values(csv_file_path1, column_index)
list2 = get_unique_column_values(csv_file_path2, column_index)
'''
unique_set = set(list1)
unique_set.update(list2)
merged_list = list(unique_set)
'''
import json

print('len',len(list2))

# Specify the path and filename for the JSON file
json_file_path = 'outputs/OKVQA_can_ans.json'

# Open the file in write mode and save the list as JSON
with open(json_file_path, 'w') as file:
    json.dump(list2, file)
    

print(begh)
from transformers import GPT2TokenizerFast, AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|endoftext|>')

#tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

tokenizer2 = AutoTokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b", use_fast=False)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 124
tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer

inputs = tokenizer("yes please", return_tensors="pt")
print(inputs['input_ids'].shape[1])


print("dir(tokenizer)",dir(tokenizer))
print("tokenizer.pad_token_id",tokenizer.pad_token_id)
#print("tokenizer.mask_token",tokenizer2.pad_token)
print("tokenizer.eos_token_id",tokenizer.eos_token_id)
print("tokenizer.cls_token_id",tokenizer.cls_token_id)
print("tokenizer.bos_token_id",tokenizer.bos_token_id)

'''
print("tokenizer.mask_token",tokenizer2.sep_token)
print("tokenizer.mask_token",tokenizer2.sep_token_id)

print("dir(tokenizer2)",dir(tokenizer2))

print("dir(tokenizer3)",dir(tokenizer3))
print("tokenizer3.pad_token_id",tokenizer3.pad_token_id)
#print("tokenizer3.mask_token",tokenizer3.pad_token)
print("tokenizer3.eos_token_id",tokenizer3.eos_token_id)
print("tokenizer3.pad_token",tokenizer3.pad_token)
print("tokenizer3.eos_token",tokenizer3.eos_token)


'''
#tokenizer2.pad_token_id = tokenizer2.eos_token
print('done1')
#tokenizer3.pad_token_id = tokenizer3.eos_token


#print("tokenizer.pad_token_id",tokenizer.model_max_length)
#print("tokenizer.eos_token_id",tokenizer.cls_token)
'''
print("tokenizer2.eos_token",tokenizer2.eos_token)
print("tokenizer2.pad_token_id",tokenizer2.pad_token_id)
print("tokenizer2.eos_token_id",tokenizer2.eos_token_id)

print("tokenizer3.eos_token",tokenizer3.eos_token)
print("tokenizer3.pad_token_id",tokenizer3.pad_token_id)
print("tokenizer3.eos_token_id",tokenizer3.eos_token_id)
'''

