# Load model directly
from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch

print("Hello")

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

device = torch.device('cuda')

model = model.to(device)

print("Hello")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# # Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
x = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(x)