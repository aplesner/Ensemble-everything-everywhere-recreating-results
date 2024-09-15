import os

PROJECT_NAME = os.getenv("PROJECT_NAME")
if PROJECT_NAME != "recreating_ensemble_everything_everywhere":
    raise ValueError("Source the project variables from helper_script/project_variables.sh before running")

DATA_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR")
