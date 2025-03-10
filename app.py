from transformers import AutoTokenizer, pipeline
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


# Define the pipeline
text_generator = pipeline("text-generation", model=checkpoint, tokenizer=tokenizer)

# Define the function to generate text
def get_response(message):
    
    system_message = {"role": "system",
                      "content": "You are a chatbot with an extreme cockney accent."
                     }
    
    message = {"role": "user",
               "content": message
              }
    
    messages = [system_message, message]
    
    output = text_generator(messages, min_new_tokens=50, max_new_tokens=200, num_beams=3, early_stopping=True)
    
    response = output[0]['generated_text'][-1]['content']
    
    return response

# Create the interface
gr.Interface(
    fn=get_response,
    inputs="text",
    outputs="text"
).launch()