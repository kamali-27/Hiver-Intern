"""
Dataset Downloader & Fallback Handler.

Attempts to acquire Kaggle dataset 'thoughtvector/customer-support-on-twitter'.
If Kaggle credentials or network access are unavailable, it seamlessly triggers
the realistic synthetic fallback generator (`data/make_sample_data.py`).
"""

import os
import sys
import subprocess
import shutil

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TARGET_FILE = os.path.join(RAW_DATA_DIR, "twcs.csv")

def ensure_dataset() -> str:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) > 1000:
        print(f"[download_data] Found existing dataset at {TARGET_FILE} ({os.path.getsize(TARGET_FILE)} bytes).")
        return TARGET_FILE

    print("[download_data] `twcs.csv` not found locally. Attempting download via Kaggle...")
    download_success = False

    # Attempt 1: kagglehub
    try:
        import kagglehub
        print("[download_data] kagglehub detected, attempting download...")
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        downloaded_csv = os.path.join(path, "twcs.csv")
        if os.path.exists(downloaded_csv):
            shutil.copy(downloaded_csv, TARGET_FILE)
            download_success = True
            print(f"[download_data] Successfully acquired dataset via kagglehub: {TARGET_FILE}")
    except Exception as e:
        print(f"[download_data] kagglehub download unavailable ({e}).")

    # Attempt 2: kaggle CLI
    if not download_success:
        try:
            print("[download_data] Checking kaggle CLI...")
            result = subprocess.run(
                ["kaggle", "datasets", "download", "-d", "thoughtvector/customer-support-on-twitter", "-p", RAW_DATA_DIR, "--unzip"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0 and os.path.exists(TARGET_FILE):
                download_success = True
                print(f"[download_data] Successfully acquired dataset via kaggle CLI: {TARGET_FILE}")
            else:
                print(f"[download_data] Kaggle CLI output: {result.stderr.strip() or result.stdout.strip()}")
        except Exception as e:
            print(f"[download_data] Kaggle CLI unavailable ({e}).")

    # Fallback to make_sample_data
    if not download_success:
        print("[download_data] NOTE: Kaggle download credentials not present or offline.")
        print("[download_data] Running automatic fallback: generating realistic synthetic dataset...")
        from data.make_sample_data import generate_synthetic_dataset
        generate_synthetic_dataset(TARGET_FILE, n_conversations=2400)
        print(f"[download_data] Fallback complete. Synthetic dataset created at {TARGET_FILE}.")

    return TARGET_FILE

if __name__ == "__main__":
    ensure_dataset()
