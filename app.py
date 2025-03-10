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