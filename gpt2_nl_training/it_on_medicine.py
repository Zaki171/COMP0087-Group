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
from nltk.translate.bleu_score import corpus_bleu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device being used:", device)

def evaluate_example(example, model, tokenizer):
    prompt = f"### Question: {example['input']} \n ### Options: {example['options']} \n ###Targets: "
    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(device)
    # print(prompt)
    outputs =model.generate(**tokenized_prompt, pad_token_id=tokenizer.eos_token_id, max_length=1024)
    decoded_output = tokenizer.decode(outputs[0][len(tokenized_prompt['input_ids'][0]):], skip_special_tokens=True)
    # print("prediction: ",decoded_output)
    return decoded_output

def evaluate_dataset(dataset, model1, model2, tok1, tok2, ngram=1):
    preds1 = []
    preds2 = []
    reals = []
    for example in dataset:
        pred1 = evaluate_example(example, model1, tok1)
        pred2 = evaluate_example(example, model2, tok2)
        correct_idx = int(example['gold_index'])
        real = example['options'][correct_idx]
        preds2.append(pred2)
        # print("plain ans: ",pred_plain)
        # print("real ans: ", real)
        # print("it ans:", pred_it)
        preds1.append(pred1)
        reals.append(real)

    P, R, F1 = bert_score.score(preds1, reals, lang="en")
    average_F1 = sum(F1) / len(F1)
    print("Average F1 score for model1:", average_F1)

    P, R, F1 = bert_score.score(preds2, reals, lang="en")
    average_F1 = sum(F1) / len(F1)
    print("Average F1 score for model2:", average_F1)

    bleu = evaluate.load('bleu')
    bleu1 = bleu.compute(predictions=preds1, references=reals, max_order=ngram)
    bleu2 = bleu.compute(predictions=preds2, references=reals, max_order=ngram)

    print("Model1 BLEU: ", bleu1)
    print("Model2 BLEU: ", bleu2)


# def calculate_bleu(preds, reals):
#     # Convert predictions and references into lists of lists of tokens
#     preds_tokens = [pred.split() for pred in preds]
#     reals_tokens = [[real.split()] for real in reals]
    
#     # Calculate BLEU score
#     bleu_score = corpus_bleu(reals_tokens, preds_tokens)
#     return bleu_score

# # Calculate BLEU score for non-IT model
# bleu_score_plain = calculate_bleu(preds_plain, reals)
# print("BLEU score for non-IT model:", bleu_score_plain)

# # Calculate BLEU score for IT model
# bleu_score_it = calculate_bleu(preds_it, reals)
# print("BLEU score for IT model:", bleu_score_it)

model_plain =  GPT2LMHeadModel.from_pretrained( "gpt2").to(device)
tokenizer_plain = GPT2Tokenizer.from_pretrained("gpt2")

model_it =  GPT2LMHeadModel.from_pretrained( "old_test/trained_model").to(device)
tokenizer_it = GPT2Tokenizer.from_pretrained("old_test/tokenizer")

mcq_dataset = load_dataset("AdaptLLM/medicine-tasks", 'MQP')['test'].select(range(50))
evaluate_dataset(mcq_dataset, model_plain, model_it, tokenizer_plain, tokenizer_it)



"""
08/03/24

mistral 
zephyr
vicuna

natural language explanations for faithfulness tests


METEOR metric
bleurt

"""