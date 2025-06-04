import gradio as gr
from huggingface_hub import login, InferenceClient, Repository
from dotenv import load_dotenv
import os
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import tempfile
import warnings
from datetime import datetime, timezone
import subprocess
import re


# -----------------------------------------------------
# Suppress warnings
# -----------------------------------------------------
warnings.filterwarnings("ignore", message=".*'Repository'.*is deprecated.*", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------
# Initialize Global Variables
# -----------------------------------------------------
global global_flagged_df
global_flagged_df = pd.DataFrame()

global message_cnt
message_cnt = 0

global inference_cnt
inference_cnt = 0

#-----------------------------------------------------
# Initialize Environment Variables
#-----------------------------------------------------
# Stopwords for tag-query matching
stopwords = {
    'and', 'or', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'with',
    'is', 'are', 'was', 'were', 'be', 'being', 'been', 'at', 'by', 'from',
    'that', 'this', 'it', 'as', 'your', 'you', 'i', 'me', 'my', 'our', 'we', 'large', 'data'
}

# -----------------------------------------------------
# Initialize Embeddings
# -----------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

### Load Q&A CSV file ###
qa_df = pd.read_csv("interview_questions_and_answers.csv")

# Precompute embeddings for the questions
qa_questions = qa_df["questions"].tolist()
qa_answers = qa_df["answers"].tolist()
qa_embeddings = embedder.encode(qa_questions, convert_to_tensor=True)

# -----------------------------------------------------
### Load the Projects Descriptions CSV file ###
projects_df = pd.read_csv("projects_table.csv")

# Precompute embeddings
project_descriptions = projects_df['description'].tolist()
project_embeddings = embedder.encode(project_descriptions, convert_to_tensor=True)


# ----------------------------------------------------
# Functions #
# ----------------------------------------------------
def update_flag_dataset():
    """
    """
    global global_flagged_df
    
    # Load token and repo info
    dataset_repo = "https://huggingface.co/datasets/im93/interview_bot_flags"
    log_file_path = "/tmp/flags/log.csv"
    hf_token = huggingface_login  # already loaded from .env or environment

    # Skip if log file doesn't exist
    if not os.path.exists(log_file_path):
        #debug#print("No flagged log file found.")
        return

    # Read the flagged data
    flagged_df = pd.read_csv(log_file_path)
    
    if flagged_df.equals(global_flagged_df): # If the global_flagged_df is equal to the flagged_df, then don't need to update
        #debug#print("No new flagged logs to update.")
        return
    
    # Create a temporary directory to clone the dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(local_dir=tmpdir, clone_from=dataset_repo, use_auth_token=hf_token)
        
        # Set Git username and email
        subprocess.run(["git", "config", "user.name", "HF Bot"], cwd=tmpdir)
        subprocess.run(["git", "config", "user.email", "bot@example.com"], cwd=tmpdir)

        
        repo.git_pull()  # ensure it's up-to-date

        dataset_file_path = os.path.join(tmpdir, "log.csv")

        

        # If the dataset file already exists, append; otherwise, create it
        if os.path.exists(dataset_file_path):
            existing_df = pd.read_csv(dataset_file_path)
            combined_df = pd.concat([existing_df, flagged_df], ignore_index=True)
        else:
            combined_df = flagged_df

        # Remove duplicates
        combined_df.drop_duplicates(subset=["flag"], keep="first", inplace=True)
        # Order by the "flag" column
        combined_df.sort_values(by=["flag"], ascending=True, inplace=True)
        # Reset the index
        combined_df.reset_index(drop=True, inplace=True)
        
        # Save updated data
        combined_df.to_csv(dataset_file_path, index=False)

        # Commit and push to the Hub
        repo.push_to_hub(commit_message="Add new flagged logs")
        
        # Update the global_flagged_df
        global_flagged_df = flagged_df
        

    print("Dataset updated successfully.")
    
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

def retrieve_relevant_projects(query, top_n=2, threshold=0.3, boost_tags=True, tag_weight=0.3):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, project_embeddings)[0]

    query_lower = query.lower()
    query_cleaned = re.sub(r"[^\w\s]", "", query_lower)
    query_words = set(word for word in query_cleaned.split() if word not in stopwords)
    
    project_scores = []

    for idx, sim in enumerate(similarities):
        score = sim.item()
        tag_score = 0
        matched_tags = []

        if boost_tags:
            tags = projects_df.iloc[idx]["tags"]
            tags_list = [t.strip() for t in tags.split(",") if t.strip()]

            for tag in tags_list:
                tag_words = set(word for word in tag.lower().split() if word not in stopwords)
                if query_words & tag_words:
                    matched_tags.append(tag)

            tag_score = tag_weight * len(matched_tags)

        total_score = score + tag_score
        project_scores.append((idx, total_score, matched_tags))

    # Sort projects by total score
    sorted_projects = sorted(project_scores, key=lambda x: x[1], reverse=True)

    # Select top N with sufficient score
    selected = []
    for idx, score, matched_tags in sorted_projects:
        if score >= threshold and len(selected) < top_n:
            selected.append({
                "title": projects_df.iloc[idx]["title"],
                "description": projects_df.iloc[idx]["description"],
                "tags": projects_df.iloc[idx]["tags"],
                "matched_tags": matched_tags,
                "link": projects_df.iloc[idx]["link"],
                "score": round(score, 3)
            })

    return selected

def add_projs_to_system_message(query):
    """
    Add the project descriptions portion of the system message
    """
    projects = retrieve_relevant_projects(query)
    
    # If no relevant projects, return
    if not projects:
        return None
    
    
    text = "\n\nHere are some relevant projects you can discuss in your answer. Use the project's description and relevancy score to help you answer the question. Be sure to supply the user with the link to the project.\n"
    for project in projects:
        text += f"\n[Project: {project['title']}"
        text += f"\nDescription: {project['description']}"
        text += f"\nRelevancy Score: {project['score']}"
        text += f"\nLink: {project['link']}]\n"
        
    return text
        
def build_rag_prompt(user_input, n_qas=3):
    system_prompt = load_system_prompt()

    # Add relevant Q&A examples
    relevant_qas = retrieve_relevant_qas(user_input, top_n=n_qas)
    system_prompt += "\n\nHere are some relevant interview Q&A examples to help guide your answers. Use them to inform how you answer and your writing voice: {"
    for i, qa in enumerate(relevant_qas):
        system_prompt += f"\n[Example {i + 1}: \nQuestion: {qa['question']}\nAnswer: {qa['answer']}]\n"

    system_prompt += "}"

    # Add relevant projects
    relevant_projects = add_projs_to_system_message(user_input)
    if relevant_projects:
        system_prompt += relevant_projects
        
    return system_prompt

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

#-------------------------------------------
# Initialize Hugging Face Client and variables
#-------------------------------------------

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
checkpoint = "meta-llama/Llama-3.3-70B-Instruct" #"google/gemma-3-27b-it"
client = InferenceClient(
    checkpoint,
    provider="nebius",
    token=huggingface_login
)

config = AutoConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

# Get the context length of the model
context_length = getattr(config, "max_position_embeddings", None)
if context_length is None:
    # If the model is google/gemma-3-27b-it, set the context length to 128000
    if checkpoint == "google/gemma-3-27b-it":
        context_length = 128000
    else:
        raise ValueError(f"Could not determine context length for model {checkpoint}")
    

#-------------------------------------------
# Define the Respond Function
#-------------------------------------------

def respond(
    message,
    history: list[dict],  # chat history
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    emergency_stop_threshold=500
):
    # Update the flagged dataset
    update_flag_dataset()
    
    # Initialize the message count
    global message_cnt
    message_cnt += 1
    
    # Initialize the inference count
    global inference_cnt
        
    # Emergency stop
    if inference_cnt > emergency_stop_threshold:
        yield "Wow, looks like this bot has been getting a lot of traffic and has exceeded my budget for computational costs. Please consider [donating to the project](https://www.paypal.com/ncp/payment/NF8NSLUCBLVUS) and try again later."
        return
    
    # Show the current time to the LLM
    now = datetime.now(timezone.utc)
    system_message = f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # Build a fresh system message using the latest user input
    system_message += build_rag_prompt(message)

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

    # Check if the user message is "dummy"
    if message.strip().lower() == "dummy":
        # Add a dummy response to the conversation history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "this is a dummy string to prevent using tokens"})
        response = "this is a dummy string to prevent using tokens"
        yield response

    else:    
        # Generate the model's response
        response = ""
        try:
            ### For Debugging - Avoids usage of tokens ###
            # response = "This is a dummy chat completion"
            # yield response
            ##############################################
            
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
                
            # Add to the inference count
            inference_cnt += 1
            
            # Print the inference count if divisible by 10
            if inference_cnt % 10 == 0:
                print(f"Inference count: {inference_cnt}")

        except Exception as e:
            print("An error occurred during chat completion:")
            print(f"Error: {e}")
            print("Messages:")
            for msg in messages:
                print(f"Role: {msg['role']}, Content: {msg['content']}")
                print()
            yield f"An error occurred during chat completion: {e}\n Refresh the page and try again."
            raise  # Re-raise the exception after logging
        
    ### Logging ###
    print("####################")
    
    # print the system message
    print(f"System Message: {messages[0]['content']}")
    
    # print the last message
    print(f"User Message: {messages[-1]['content']}")
    print(f"response: {response}")
    
    

#-------------------------------------------
### Define the chat interface ###
#-------------------------------------------

demo = gr.ChatInterface(
    respond,
    type="messages",
    title="Isaiah Montoya Instant Interview",
    description="""
<div style="text-align: center;">
Ask me anything about my experience.

Try copy/pasting a job description to talk about a specific role.

Connect with me on [Linkedin](https://www.linkedin.com/in/isaiah-montoya/). <br>
Checkout the [Github Repo](https://github.com/IMMontoya/instant_interview_bot).
</form>
</div>
""",

############### DONATION BUTTON #################### 
# Consider [donating](https://www.paypal.com/ncp/payment/NF8NSLUCBLVUS) to support this project:

# <style>.pp-NF8NSLUCBLVUS{text-align:center;border:none;border-radius:0.25rem;min-width:11.625rem;padding:0 2rem;height:2.625rem;font-weight:bold;background-color:#FFD140;color:#000000;font-family:"Helvetica Neue",Arial,sans-serif;font-size:1rem;line-height:1.25rem;cursor:pointer;}</style>
# <form action="https://www.paypal.com/ncp/payment/NF8NSLUCBLVUS" method="post" target="_blank" style="display:inline-grid;justify-items:center;align-content:start;gap:0.5rem;">
#   <input class="pp-NF8NSLUCBLVUS" type="submit" value="Donate via PayPal" style="font-size: 1.5rem; color: black; background-color: yellow; border-radius: 0.25rem;" />
#   <img src="https://www.paypalobjects.com/images/Debit_Credit_APM.svg" alt="cards" />
  
    chatbot=gr.Chatbot(placeholder="<div style='text-align: center;'>This is a chatbot designed to provide instant interview responses as if it were me (Isaiah Montoya).</div>",
    type="messages"),
    flagging_mode="manual",
    flagging_dir="/tmp/flags",
    stop_btn=False,
    editable=True,
)

### Launch the Interface ###
if __name__ == "__main__":
    demo.launch()