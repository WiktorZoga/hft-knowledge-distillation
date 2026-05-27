import os
import sys
import yaml
import zipfile
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
import mlcroissant as mlc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

ENV_PATH = PROJECT_ROOT / ".env"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TARGET_DATA_DIR = PROJECT_ROOT / "datasets"

TARGET_DATA_DIR.mkdir(exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(TARGET_DATA_DIR)

load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("KAGGLE_API_TOKEN"):
    raise ValueError("CRITICAL ERROR: 'KAGGLE_API_TOKEN' missing inside local .env file.")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file missing: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

croissant_url = config["data"]["croissant_url"]

print(f"Streaming dataset via MLCroissant endpoint into: {TARGET_DATA_DIR}")

try:
    # Disable strict validation warnings crashing the script
    dataset = mlc.Dataset(croissant_url, mapping={})
    
    # Extract file URLs from the distribution metadata
    distributions = dataset.metadata.distribution
    if not distributions:
        raise RuntimeError("No file distributions found in Croissant metadata.")
    
    # Find the zip file download URL
    download_url = None
    for dist in distributions:
        content_url = getattr(dist, "content_url", "")
        if "download" in content_url or content_url.endswith(".zip"):
            download_url = content_url
            break
            
    if not download_url:
        # Fallback to the first available content URL if no explicit zip is found
        download_url = distributions[0].content_url

    zip_target_path = TARGET_DATA_DIR / "fi2010_raw.zip"
    
    print(f"Fetching archive payload from: {download_url}")
    
    # Configure urllib with basic authentication using your token
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0'),
        ('Authorization', f"Bearer {os.getenv('KAGGLE_API_TOKEN')}")
    ]
    urllib.request.install_opener(opener)
    
    urllib.request.urlretrieve(download_url, zip_target_path)
    print("Download successful. Extracting packet archives locally...")
    
    with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DATA_DIR)
        
    os.remove(zip_target_path)
    print("Extraction complete. Cleaned temporary zip archives.")
    print("\nPIPELINE INFRASTRUCTURE READINESS ACHIEVED")

except Exception as e:
    raise RuntimeError(f"CRITICAL ERROR during Croissant execution pipeline: {e}")