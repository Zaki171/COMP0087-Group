import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from encodeinstruction import encodeinstruction
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import json
import bert_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
import evaluate


torch.cuda.empty_cache()
a=torch.cuda.FloatTensor()
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

output_dir_model = "new_test/trained_model"
output_dir_tokenizer = "new_test/tokenizer"

local_dataset = load_from_disk("data/100_per_task")
local_dataset['test'] = local_dataset['test'].select(range(100))



def formatting_func(examples):
    # outputs = []
    # for i in range(len(examples['task_name'])):
    text = f"### Task: {examples['definition']}\n ### Inputs: {examples['inputs']} \n ### Targets: "
    # text = tokenizer(text, max_length=1024)
        # outputs.append(text)
    return text

def generate(prompt, tokenizer, model):
        tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            output = model.generate(**tokenized_prompt, max_length=1024, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][len(tokenized_prompt['input_ids'][0]):], skip_special_tokens=True)


model_trained =  GPT2LMHeadModel.from_pretrained(output_dir_model).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir_tokenizer)




predictions = []
real_vals = []
for eg in local_dataset['test']:
     prompt = formatting_func(eg)
     pred = generate(prompt, tokenizer, model_trained)
     truth = eg['targets']
     predictions.append(pred)
     real_vals.append(truth)

print("done")
P, R, F1 = bert_score.score(predictions, real_vals, lang="en")
average_F1 = sum(F1) / len(F1)
print("Average F1 score:", average_F1)

bleu = evaluate.load('bleu')
bleu1 = bleu.compute(predictions=predictions, references=real_vals)
print(bleu1)

bleurt = evaluate.load('bleurt',  checkpoint='BLEURT-20')
bleurt1 = bleurt.compute(predictions=predictions, references=real_vals)
print(sum(bleurt1['scores'])/len(bleurt1['scores']))