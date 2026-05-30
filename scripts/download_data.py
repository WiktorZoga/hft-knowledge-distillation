import shutil
from pathlib import Path
import kagglehub

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TARGET_DATA_DIR = PROJECT_ROOT / "datasets"
TARGET_DATA_DIR.mkdir(exist_ok=True)

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
