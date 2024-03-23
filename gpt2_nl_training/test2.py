from transformers import AutoModelForCausalLM, AutoTokenizer


max_num_egs = 3 #natural instructions are just too big

# model_plain =  GPT2LMHeadModel.from_pretrained("gpt2").to(device)
# tokenizer_plain = GPT2Tokenizer.from_pretrained("gpt2")
# print("models retrieved") 



model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")


def evaluate_example2(model, tokenizer, prompt):
    # model_inputs = tokenizer(prompt, return_tensors="pt")
    # print(prompt)
    # if len(tokenized_prompt['input_ids'][0]) > MAX_LENGTH: #currently just checking if random prompt is too big or not
    #     return None 
    messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    # outputs =model.generate(**tokenized_prompt, pad_token_id=tokenizer.eos_token_id, max_length=8000)
    # decoded_output = tokenizer.decode(outputs[0][len(tokenized_prompt['input_ids'][0]):], skip_special_tokens=True)
    # # print("prediction: ",decoded_output)
    # return decoded_output


prompt = "My favourite condiment is"
pred = evaluate_example2(prompt, model, tokenizer) 