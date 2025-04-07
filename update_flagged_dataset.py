from huggingface_hub import HfApi, Repository, login
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv

### Login to the Hugging Face Hub ###
load_dotenv()
huggingface_login = os.getenv("HUGGINGFACE_LOGIN")
if huggingface_login is None:
    try: # For loading from the huggingface space
        huggingface_login = os.environ.get("HUGGINGFACE_LOGIN")
    except:
        raise ValueError("Please provide your Hugging Face login credentials.")

login(token=huggingface_login, add_to_git_credential=True)

def update_flag_dataset():
    # Load token and repo info
    dataset_repo = "https://huggingface.co/datasets/im93/interview_bot_flags"
    log_file_path = "/tmp/flagged/log.csv"
    hf_token = huggingface_login  # already loaded from .env or environment

    # Skip if log file doesn't exist
    if not os.path.exists(log_file_path):
        print("No flagged log file found.")
        return

    # Create a temporary directory to clone the dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Repository(local_dir=tmpdir, clone_from=dataset_repo, use_auth_token=hf_token)
        repo.git_pull()  # ensure it's up-to-date

        dataset_file_path = os.path.join(tmpdir, "log.csv")

        # Read the flagged data
        flagged_df = pd.read_csv(log_file_path)

        # If the dataset file already exists, append; otherwise, create it
        if os.path.exists(dataset_file_path):
            existing_df = pd.read_csv(dataset_file_path)
            combined_df = pd.concat([existing_df, flagged_df], ignore_index=True)
        else:
            combined_df = flagged_df

        # Remove duplicates
        combined_df.drop_duplicates(subset=["flag"], keep="first", inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        
        # Save updated data
        combined_df.to_csv(dataset_file_path, index=False)

        # Commit and push to the Hub
        repo.push_to_hub(commit_message="Add new flagged logs")

    print("Dataset updated successfully.")

# Run when script is executed
if __name__ == "__main__":
    update_flag_dataset()