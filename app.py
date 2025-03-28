
import gradio as gr
from huggingface_hub import login, InferenceClient
from dotenv import load_dotenv
import os
import pandas as pd
from transformers import AutoConfig, AutoTokenizer

### Define Functions ###
#### Loading the System Message ####
def load_system_prompt():
    system_prompt = ""
    with open("system_prompt.txt", "r") as f:
        system_prompt = f.read()
    
    # Replace "//RESUME//" with the resume text
    with open("resume.txt", "r") as f:
        system_prompt = system_prompt.replace("//RESUME//", f.read())
        
    return system_prompt

def add_examples(system_prompt, n_examples):
    # assert n_examples > 0
    assert n_examples >= 0, "n_examples must be greater than 0"
    
    if n_examples == 0:
        return system_prompt
    
    # Read the "interview_questions_and_answers.csv" file
    df = pd.read_csv("interview_questions_and_answers.csv")
    
    # If n_examples is greater than the number of rows in the DataFrame, set n_examples to the number of rows
    if n_examples > len(df):
        n_examples = len(df)
        print(f"n_examples is greater than the number of rows in the DataFrame. Setting n_examples to {n_examples}.")
    
    # Add context to the system prompt
    system_prompt += "\n\nHere are some examples of questions you might be asked and how you should answer them:{"
    
    # Get the first n_examples rows
    examples = df.head(n_examples)
    
    # Add each example to the system prompt
    for index, row in examples.iterrows():
        system_prompt += "\n\nExample " + str(index + 1) + ": " "{" + "\nQuestion: {" + row["questions"] + "}" + "\nAnswer: {" + row["answers"] + "}" + "\n}"
        
    # Close the system prompt
    system_prompt += "\n}"
    
    return system_prompt

def build_system_message(n_examples, projects=True):
    system_prompt = load_system_prompt()
    system_prompt = add_examples(system_prompt, n_examples)
    
    if projects:
        with open("projects.txt", "r") as f:
            system_prompt += "\n\n" + f.read()
            
    return system_prompt

#### Load the System Message ####
system_message = build_system_message(5, projects=True)

### Login to the Hugging Face Hub ###
load_dotenv()
huggingface_login = os.getenv("HUGGINGFACE_LOGIN")
if huggingface_login is None:
    try: # For loading from the huggingface space
        huggingface_login = os.environ.get("HUGGINGFACE_LOGIN")
    except:
        raise ValueError("Please provide your Hugging Face login credentials.")

login(token=huggingface_login, add_to_git_credential=True)

### Define the Model and client ###
checkpoint = "google/gemma-3-27b-it"
client = InferenceClient(
    checkpoint,
    token=huggingface_login
)

config = AutoConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

# Get the context length of the model
context_length = getattr(config, "max_position_embeddings", None)
if context_length is None:
    # If the model is google/gemma-3-27b-it, set the context length to 128000
    if checkpoint == "google/gemma-3-27b-it":
        context_length = 6000
    else:
        raise ValueError(f"Could not determine context length for model {checkpoint}")

### Define the summarize_message function ###
def summarize_message(content):
    summary_system_message = "You are a bot programmed to summarize chatbot messages. Speak as directly as possible with no unnecessary words. You respond by providing a summary of the user's message as if it was written by the user."
    
    messages = [{"role": "system", "content": summary_system_message},
                {"role": "user", "content": content}]
    
    response = client.chat_completion(
        messages,
        stream=False,
        seed=42,
        max_tokens=200
    )
    
    return response.choices[0].message.content

### Define the response function ###
def respond(
    message,
    history: list[dict],  # Updated to reflect the new format
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    system_message=system_message,
):
    messages = [{"role": "system", "content": system_message}]

    for msg in history:
        messages.append(msg)  # Directly append the OpenAI-style messages

    messages.append({"role": "user", "content": message})
    
    # Tokenize the messages and count the tokens
    combined_messages = " ".join([msg["content"] for msg in messages])  # Combine all the message contents
    tokenized_input = tokenizer(combined_messages, return_tensors="pt", truncation=False, padding=False)
    
    # Get token length
    token_length = tokenized_input.input_ids.shape[1] + max_tokens  # Add the max_tokens to the token length
    print(f"Max tokens: {max_tokens}")
    print(f"Token length: {token_length}")
    print(f"Context length: {context_length}")
    
    truncated = False
    # Summarize the message if token length is greater than the context length
    while token_length > context_length:
        summarized = False
        # Find the first message (excluding the system message)
        for i, msg in enumerate(messages[1:]):
            message_tokens = tokenizer(msg["content"], return_tensors="pt", truncation=False, padding=False)
            if message_tokens.input_ids.shape[1] > 200:
                truncated = True
                summarized = True
                # Summarize the message
                msg["content"] = summarize_message(msg["content"])
                break
            
        if not summarized:
            # Remove the first user, assistant pair
            messages.pop(1)
            messages.pop(1)
            truncated = True
                
        # Recalculate the token length
        combined_messages = " ".join([msg["content"] for msg in messages])  # Combine all the message contents
        tokenized_input = tokenizer(combined_messages, return_tensors="pt", truncation=False, padding=False)
        token_length = tokenized_input.input_ids.shape[1] + max_tokens  # Add the max_tokens to the token length
            
    if truncated:
        print("History roles:")
        for message in messages:
            print(f"{message['role']}: {message['content'][:50]}")

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


### Define the Interface ###
demo = gr.ChatInterface(
    respond,
    type="messages",
    title="Isaiah Montoya Instant Interview",
    description="Connect with me on [Linkedin](https://www.linkedin.com/in/isaiah-montoya/).",
    chatbot=gr.Chatbot(placeholder="This is a chatbot designed to provide instant interview responses as if it were me (Isaiah Montoya).",
                       type="messages"),
    flagging_mode="manual",
)

### Launch the Interface ###
if __name__ == "__main__":
    demo.launch()