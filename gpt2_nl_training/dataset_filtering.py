import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from encodeinstruction import encodeinstruction
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json
import pandas as pd

# torch.cuda.empty_cache()
# a=torch.cuda.FloatTensor()
# print(torch.cuda.is_available())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device being used:", device)

# output_dir = "./trained_model"
# output_tokenizer = "./tokenizer"

from collections import Counter

local_dataset = load_from_disk("data/filtered_instr")

# Grouping by task_name and counting occurrences


def filter(ds):
    pandas_dataset = ds.to_pandas()
    # print(pandas_dataset.head())
    # print(len(pandas_dataset))
    dfs = []
    counter=0
    for task_name, group in pandas_dataset.groupby('task_name'):
        counter+=1
        dfs.append(group.drop_duplicates().head(1000))
    print(counter)
    concatenated_df = pd.concat(dfs, ignore_index=True)
    new_dataset = Dataset.from_pandas(concatenated_df)
    return new_dataset

train_dataset = filter(local_dataset['train'])
val_dataset = filter(local_dataset['validation'])
new_ds_dict = {
    'train': train_dataset,
    'validation': val_dataset,
    'test': local_dataset['test']
}

# Create a DatasetDict object
dataset_dict = DatasetDict(new_ds_dict)

# Save the DatasetDict object to disk
dataset_dict.save_to_disk("data/1000_per_task")