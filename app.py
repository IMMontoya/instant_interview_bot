import torch
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import gradio as gr
from huggingface_hub import login
from dotenv import load_dotenv
import os


# Log in to the Hugging Face Hub
load_dotenv()
huggingface_login = os.getenv("HUGGINGFACE_LOGIN")
if huggingface_login is None:
    try:
        huggingface_login = os.environ.get("HUGGINGFACE_LOGIN")
    except:
        raise ValueError("Please provide your Hugging Face login credentials.")

login(token=huggingface_login, add_to_git_credential=True)

# Load the model and tokenizer
checkpoint = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Ensure a pad token is set (LLaMA models often don't have one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")

# Custom Logits Processor for Presence and Frequency Penalties
class PresenceFrequencyPenaltyProcessor(LogitsProcessor):
    def __init__(self, presence_penalty=0.0, frequency_penalty=0.0):
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def __call__(self, input_ids, scores):
        token_counts = torch.bincount(input_ids[0], minlength=scores.shape[-1]).float()

        # Apply presence penalty (penalizes existing tokens)
        presence_mask = (token_counts > 0).float()
        scores -= self.presence_penalty * presence_mask

        # Apply frequency penalty (penalizes frequently occurring tokens)
        scores -= self.frequency_penalty * token_counts

        return scores
    

# Custom Logits Processor for Presence and Frequency Penalties
class PresenceFrequencyPenaltyProcessor:
    def __init__(self, presence_penalty=0.0, frequency_penalty=0.0):
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def __call__(self, input_ids, scores):
        token_counts = torch.bincount(input_ids[0], minlength=scores.shape[-1]).float()
        presence_mask = (token_counts > 0).float()
        scores -= self.presence_penalty * presence_mask
        scores -= self.frequency_penalty * token_counts
        return scores

# Read the few-shot interview prompt
with open("few_shot_interview_prompt.txt") as f:
    system_message = f.read()
    
# Select the number of examples for the few-shot learning
def select_n_few_shot_examples(n, system_message):
    
    # Split the system message by lines
    system_message_lines = system_message.split("\n")
    new_system_message_lines = []
    # Find the line that starts with "Example [n+1]:"
    for i, line in enumerate(system_message_lines):
        if line.startswith(f"Example {n+1}:"):
            break
        new_system_message_lines.append(line)
        
    # The last line is closes the system message
    new_system_message_lines.append(system_message_lines[-1])
    system_message = "\n".join(new_system_message_lines)
    
    return system_message

system_message = select_n_few_shot_examples(3, system_message)
    
# Replace "//RESUME//" with the resume text
with open("resume.txt") as f:
    system_message = system_message.replace("//RESUME//", f.read())
    
# Print the number of tokes in the system message
print(f"Number of tokens in the system message: {len(tokenizer(system_message)['input_ids'])}")

# Print the max token length of the model
print(f"Max token length of the model: {tokenizer.model_max_length}")


# Function to generate text using chat-style formatting
def generate_text(user_message, system_message=system_message, presence_penalty=0.6, frequency_penalty=0.4, max_new_tokens=200, temperature=0.8, top_p=0.9):
    
    # Chat format for LLaMA models
    formatted_prompt = f"<|start|>system\n{system_message}<|end|>\n<|start|>user\n{user_message}<|end|>\n<|start|>assistant\n"

    # Tokenize with padding and attention mask
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,  # Ensures padding tokens are added if needed
        truncation=True
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    logits_processor = LogitsProcessorList([
        PresenceFrequencyPenaltyProcessor(presence_penalty, frequency_penalty)
    ])

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        logits_processor=logits_processor,
        temperature=temperature,
        top_p=top_p,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)




# Create the interface
gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text"
).launch()