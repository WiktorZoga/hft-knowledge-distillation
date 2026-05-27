import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import kagglehub

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ENV_PATH = PROJECT_ROOT / ".env"
TARGET_DATA_DIR = PROJECT_ROOT / "datasets"

TARGET_DATA_DIR.mkdir(exist_ok=True)

load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
    raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set in your .env file.")

print(f"Downloading FI2010 dataset via Kaggle API into: {TARGET_DATA_DIR}")

path = kagglehub.dataset_download("freemanone/fi2010")

print(f"Dataset downloaded to cache: {path}")
print("Copying files into datasets/ directory...")

for file in Path(path).rglob("*"):
    if file.is_file():
        dest = TARGET_DATA_DIR / file.name
        shutil.copy2(file, dest)
        print(f"  Copied: {file.name}")

print("\nDataset ready.")
