# Imports
import os
from pathlib import Path
import logging

# Define Logging Format
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# specify files to create
list_of_files = [
    ".env", # Local development of ENVIRONMENT vatiables
    ".github/workflows/.gitkeep", # temp file for github CICD
    # "experiments/testing.py", # for testing files if needed
    "requirements.train.txt", # for package versions for train
    "requirements.inference.txt", # for package versions for train
    "Dockerfile.training", # need training docker file
    "Dockerfile.inference", # need inferenc docker file
    "tests/__init__.py", # creation of testing folder
    "src/__init__.py", # Main Code Folder
    "src/data/__init__.py", # folder for data functions
    "src/model/__init__.py", # folder for main modeling code
    "src/inference/__init__.py", # folder for inferencing code
    "src/drift/__init__.py", # folder for code related to drift detection
]

# Go through list and create folders/files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # logic for directory
    if filedir != "":
        # create directory
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Create the files
    # check if file exists I do not want to replace
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else: # if file already exists
        logging.info(f"{filename} already exists")