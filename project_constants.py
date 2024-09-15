import os

PROJECT_NAME = os.getenv("PROJECT_NAME")
if PROJECT_NAME != "Ensemble-everything-everywhere-recreating-results":
    raise ValueError("Source the project variables from helper_script/project_variables.sh before running")

DATA_STORAGE_DIR = os.getenv("DATA_STORAGE_DIR")
