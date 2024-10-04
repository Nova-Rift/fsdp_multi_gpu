import json
import os
os.environ['HF_HOME'] = '/workspace/persistent'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = 'True'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
import bitsandbytes

seed = 100
torch.manual_seed(seed)

# Log in to Hugging Face
login(token="HF_TOKEN_PLACEHOLDER")

# Check if a GPU is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configuration for 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Optional: Use double quantization
    bnb_4bit_quant_type="nf4",  # Use NormalFloat4 (NF4) quantization type
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for compute
    bnb_4bit_quant_storage=torch.float16  # Quantization storage in float16
)

# Load the base model with quantization
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# Load the LoRA model on top of the base model
lora_model = PeftModel.from_pretrained(base_model, "llama-sft-lora-fsdp", device_map="auto").to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add the special pad token
tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})

# Input text including all the previously commented-out lines
eval_questions = [
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Does the Jayco Eagle have a queen-size bed?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>How many queen-size beds are in the Keystone Cougar?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs come with a queen-size bed?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>What is the MSRP of the Jayco Eagle?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>How much does the Keystone Cougar cost?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>What RVs have an MSRP of less than $10,000?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs are priced under $15,000 and can sleep more than 6 people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>How many people can sleep in the Jayco Eagle?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>What is the maximum sleeping capacity of the Keystone Cougar?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs can sleep 6 or more people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs can sleep 9 or more people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Does the Fleetwood Fiesta come with a shower?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Is there a standard shower in the Coachmen Clipper Sport?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs come with showers?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs have both a queen-size bed and a shower?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>What are the differences between the Jayco Eagle and Keystone Cougar?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RV is more expensive: Jayco Eagle or Fleetwood Fiesta?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs do you carry that cost less than $15,000 and sleep at least 8 people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Are there any RVs under $7,000 that include a shower?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs in your inventory are priced under $30,000 and sleep more than 5 people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Are there any RVs with a queen-size bed, priced less than $10,000?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Which RVs are available for large families?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>What is the most affordable RV that can sleep at least 6 people?<|eot_id|>',
    '<|start_header_id|>system<|end_header_id|>You are an AI assistant built by demo company. You provide simple information about four RVs: Jayco Eagle, Keystone Cougar, Coachmen Clipper, and the Fleetwood Fiesta. Keep your answer short and to the point.<|eot_id|><|start_header_id|>user<|end_header_id|>Do any RVs have more than one queen-size bed?<|eot_id|>',
]

# List to store the outputs in a format suitable for JSON
results = []

# Loop through each question, generate a response, and store the result
for idx, question in enumerate(eval_questions, start=1):
    # Tokenize and move the input to GPU
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # Generate text
    outputs = lora_model.generate(**inputs, max_new_tokens=100, temperature=0.1, top_p=0.1)

    # Decode the output and append to results list
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    results.append({"id": idx, "message": decoded_output})

# Save the results to a JSON file
with open('output_responses.json', 'w') as f:
    json.dump({"messages": results}, f, indent=4)

print("Responses have been written to output_responses.json")
