import subprocess

def scheduled_job():
    subprocess.run(["python", "update_flagged_dataset.py"])
