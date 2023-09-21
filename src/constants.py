from pathlib import Path
import os
# Get the path of the current script
script_path = Path(__file__).resolve().parent

# Calculate paths based on the script directory using Pathlib
DATA_PATH = script_path.parent / "data"
DOCUMENT_PATHS = [DATA_PATH / doc for doc in os.listdir("./data")]
