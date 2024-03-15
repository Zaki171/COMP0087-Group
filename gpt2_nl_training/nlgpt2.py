import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from encodeinstruction import encodeinstruction
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json

torch.cuda.empty_cache()
a=torch.cuda.FloatTensor()
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

output_dir = "new_test2/trained_model"
output_tokenizer = "new_test2/tokenizer"

local_dataset = load_from_disk("data/1000_per_task")
print(local_dataset.keys())
print(len(local_dataset['train']))

MAX_LENGTH = 1024

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', max_length=MAX_LENGTH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# # # dataset = load_dataset("Muennighoff/natural-instructions")


# # # dataset.save_to_disk("data/natural_instructions")





def remove_long_samples(example):
    inp = f"{example['definition']}\n ### Inputs: {example['inputs']} \n ### Targets: {example['targets']}"
    encoded_inp = tokenizer.encode(inp, return_tensors = 'pt')
    if encoded_inp.size(1) <= MAX_LENGTH:
        return True
    return False
    
# local_dataset['train'] = local_dataset['train'].filter(remove_long_samples)
# local_dataset.save_to_disk('data/filtered_instr')

def formatting_func(examples):
    # outputs = []
    # for i in range(len(examples['task_name'])):
    text = f"### Task: {examples['definition']}\n ### Inputs: {examples['inputs']} \n ### Targets: {examples['targets']}"
    # text = tokenizer(text, max_length=1024)
        # outputs.append(text)
    return text


# num_train_samples=10
# local_dataset['train'] = local_dataset['train'].select(range(num_train_samples))

# formatted_data = local_dataset['train'].map(formatting_func)
# print(formatted_data[0])

response_template = "\n ### Targets:"

# # Data collator for completion-only language modeling
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer) #collator improves padding in batches

training_args = TrainingArguments(
    output_dir='./results2',
    overwrite_output_dir=True,
    num_train_epochs=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    per_device_train_batch_size=1,
)


trainer = SFTTrainer(
    model=model,
    
    args = training_args,
    train_dataset=local_dataset['train'],
    eval_dataset = local_dataset['validation'],
    formatting_func=formatting_func,
    packing=True,
) #packing only used with dataset_text_field - pack samples together
#if tokenizer not specified then automatically chooses



trainer.train()


trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_tokenizer)
print("Model, tokenizer saved")


def generate(prompt, tokenizer, model):
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = model.generate(**tokenized_prompt, max_length=1024)
        return tokenizer.decode(output[0][len(tokenized_prompt['input_ids'][0]):], skip_special_tokens=True)


model_trained =  GPT2LMHeadModel.from_pretrained(output_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(output_tokenizer)
test_prompt = local_dataset['test'][690] #5, 700
test_prompt_text = f"### Task: {test_prompt['definition']}\n ### Inputs: {test_prompt['inputs']}\n ### Targets:"

print(test_prompt_text)



print("Generated text:")
print(generate(test_prompt_text, tokenizer, model_trained))

#old data length before removing dups: 75464
#new:  75212
