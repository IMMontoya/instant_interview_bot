import gradio as gr
from huggingface_hub import login, InferenceClient
from dotenv import load_dotenv
import os
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import re

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
    
### Load Q&A CSV file ###
qa_df = pd.read_csv("interview_questions_and_answers.csv")

# Load sentence transformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for the questions
qa_questions = qa_df["questions"].tolist()
qa_answers = qa_df["answers"].tolist()
qa_embeddings = embedder.encode(qa_questions, convert_to_tensor=True)

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



def retrieve_relevant_qas(query, top_n=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, qa_embeddings)[0]
    
    top_indices = torch.topk(similarities, k=top_n).indices.tolist()
    
    retrieved_qas = []
    for idx in top_indices:
        retrieved_qas.append({
            "question": qa_questions[idx],
            "answer": qa_answers[idx]
        })
    
    return retrieved_qas


# Load and parse the project table from projects.txt
with open("projects.txt", "r") as f:
    projects_raw = f.read()

# Extract rows from the markdown table
project_lines = [line.strip() for line in projects_raw.split("\n") if "|" in line and "Project number" not in line and "---" not in line]
project_data = []

for line in project_lines:
    parts = [part.strip() for part in line.split("|") if part.strip()]
    if len(parts) == 3:
        project_data.append({
            "number": parts[0],
            "title": parts[1],
            "description": parts[2]
        })

# Embed project descriptions
project_descriptions = [p["description"] for p in project_data]
project_embeddings = embedder.encode(project_descriptions, convert_to_tensor=True)

def retrieve_relevant_projects(query, top_n=3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, project_embeddings)[0]

    top_indices = torch.topk(similarities, k=top_n).indices.tolist()

    return [project_data[i] for i in top_indices]

def build_rag_prompt(user_input, n_qas=3, n_projects=3, projects=True):
    system_prompt = load_system_prompt()

    # Add relevant Q&A examples
    relevant_qas = retrieve_relevant_qas(user_input, top_n=n_qas)
    system_prompt += "\n\nHere are some relevant interview Q&A examples to help guide your answers:\n"
    for i, qa in enumerate(relevant_qas):
        system_prompt += f"\nExample {i + 1}:\nQuestion: {qa['question']}\nAnswer: {qa['answer']}\n"

    # Add relevant projects
    if projects:
        system_prompt += '\n\nYou can also include links to your projects ONLY WHEN THEY ARE RELEVANT. Use the "Description" column in the table to determine if the project is relevant. DO NOT WRAP THE LINKS IN ANY ADDITIONAL CHARACTERS IN YOUR RESPONSES. Here is the complete list of relevant projects:\n{'

        relevant_projects = retrieve_relevant_projects(user_input, top_n=n_projects)

        for proj in relevant_projects:
            system_prompt += f'\nProject {proj["number"]}: {proj["title"]}\nDescription: {proj["description"]}\n'

        system_prompt += "}"

    return system_prompt



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
    history: list[dict],  # chat history
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
):
    # Build a fresh system message using the latest user input
    system_message = build_rag_prompt(message, projects=True)

    # Construct the messages list
    messages = [{"role": "system", "content": system_message}]

    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": message})
    
    # Tokenize the combined messages
    combined_messages = " ".join([msg["content"] for msg in messages])
    tokenized_input = tokenizer(combined_messages, return_tensors="pt", truncation=False, padding=False)

    # Get total token length
    token_length = tokenized_input.input_ids.shape[1] + max_tokens

    # Compress or trim history if token length exceeds limit
    while token_length > context_length:
        summarized = False
        for i, msg in enumerate(messages[1:]):  # Skip system message
            message_tokens = tokenizer(msg["content"], return_tensors="pt", truncation=False, padding=False)
            if message_tokens.input_ids.shape[1] > 200:
                summarized = True
                msg["content"] = summarize_message(msg["content"])
                break
        
        if not summarized:
            # Remove the oldest user-assistant message pair
            if len(messages) > 3:
                messages.pop(1)
                messages.pop(1)
            else:
                break

        # Recalculate token length
        combined_messages = " ".join([msg["content"] for msg in messages])
        tokenized_input = tokenizer(combined_messages, return_tensors="pt", truncation=False, padding=False)
        token_length = tokenized_input.input_ids.shape[1] + max_tokens

    # Generate the model's response
    response = ""
    try:
        for output in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            token = output.choices[0].delta.content
            response += token
            yield response

    except Exception as e:
        print("An error occurred during chat completion:")
        print(f"Error: {e}")
        print("Messages:")
        print(messages)
        raise

### Define the Interface ###
demo = gr.ChatInterface(
    respond,
    type="messages",
    title="Isaiah Montoya Instant Interview",
    description="Ask me anything about my experience.\n\n*Hint: You can copy/paste a job description to talk about a specific role.* \n\nConnect with me on [Linkedin](https://www.linkedin.com/in/isaiah-montoya/).",
    chatbot=gr.Chatbot(placeholder="This is a chatbot designed to provide instant interview responses as if it were me (Isaiah Montoya).",
                       type="messages"),
    flagging_mode="manual",
        stop_btn=False,
    editable=True,
)

### Launch the Interface ###
if __name__ == "__main__":
    demo.launch()