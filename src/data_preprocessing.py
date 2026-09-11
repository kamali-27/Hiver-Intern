"""
Data Preprocessing Module for Customer Support Tweets.

1. Ensures raw dataset exists (or triggers fallback).
2. Filters to inbound customer tweets and their paired brand replies.
3. Selects 'AmazonHelp' (largest volume, broad multi-intent distribution).
4. Cleans text: normalizes whitespace, strips URLs and extraneous @handles,
   while preserving original text for UI display and customer context.
5. Saves processed conversations to `data/processed/brand_conversations.csv`.
"""

import os
import re
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "twcs.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_CONVERSATIONS_PATH = os.path.join(PROCESSED_DATA_DIR, "brand_conversations.csv")

TARGET_BRAND = "AmazonHelp"


def clean_text_for_modeling(text: str) -> str:
    """
    Cleans customer tweet text for ML vectorization:
    - Removes URLs (http/https/t.co)
    - Removes specific numeric user handles (e.g. @115712) and brand tags
    - Replaces order IDs and currency amounts with generic tokens for better generalization
    - Normalizes punctuation and excessive whitespace
    - Lowercases text
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+|t\.co/\S+|amzn\.to/\S+", " ", text)

    # Normalize order IDs (e.g. 112-9213812-1923812 or #123456)
    cleaned = re.sub(r"#?\b\d{3}-\d{7}-\d{7}\b", " order_number ", cleaned)
    cleaned = re.sub(r"#\d{4,10}\b", " order_number ", cleaned)

    # Normalize currency amounts (e.g. $49.99, $120)
    cleaned = re.sub(r"\$\d+(?:\.\d{2})?", " currency_amount ", cleaned)

    # Remove @handles
    cleaned = re.sub(r"@\w+", " ", cleaned)

    # Remove non-alphanumeric characters except basic punctuation
    cleaned = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", cleaned)

    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.lower()


# Export clean_text alias
clean_text = clean_text_for_modeling


def load_and_preprocess(raw_path: str = RAW_DATA_PATH, brand: str = TARGET_BRAND) -> pd.DataFrame:
    """
    Loads raw twcs.csv, pairs customer queries with brand replies, cleans text,
    and returns a clean DataFrame.
    """
    if not os.path.exists(raw_path):
        print(f"[preprocessing] {raw_path} not found. Triggering download/fallback generator...")
        from data.download_data import ensure_dataset
        raw_path = ensure_dataset()

    print(f"[preprocessing] Loading raw data from {raw_path}...")
    # Load raw data; twcs.csv has mixed types or missing values in some columns
    df = pd.read_csv(raw_path, low_memory=False)

    print(f"[preprocessing] Loaded {len(df):,} total raw tweets.")

    # Filter inbound customer tweets and brand outbound tweets
    inbound_df = df[df["inbound"] == True].copy()
    outbound_df = df[df["inbound"] == False].copy()

    # Normalize IDs to string for reliable joining
    inbound_df["tweet_id_str"] = inbound_df["tweet_id"].astype(str)
    outbound_df["in_response_to_tweet_id_str"] = outbound_df["in_response_to_tweet_id"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Filter outbound tweets for the selected brand
    brand_replies = outbound_df[outbound_df["author_id"].astype(str).str.lower() == brand.lower()].copy()

    if len(brand_replies) == 0:
        # Fallback: if brand not found (e.g. dataset only has another brand), pick the top brand
        top_brand = outbound_df["author_id"].value_counts().index[0]
        print(f"[preprocessing] Brand '{brand}' not found. Selecting top available brand: '{top_brand}'")
        brand = top_brand
        brand_replies = outbound_df[outbound_df["author_id"] == brand].copy()

    print(f"[preprocessing] Found {len(brand_replies):,} replies from brand '{brand}'.")

    # Join inbound customer tweets with brand reply
    # Customer tweet's tweet_id matches brand reply's in_response_to_tweet_id
    merged = pd.merge(
        inbound_df,
        brand_replies,
        left_on="tweet_id_str",
        right_on="in_response_to_tweet_id_str",
        suffixes=("_customer", "_brand")
    )

    print(f"[preprocessing] Successfully paired {len(merged):,} customer-brand conversation pairs.")

    # Deduplicate in case of multiple replies to the same customer tweet
    merged = merged.drop_duplicates(subset=["tweet_id_customer"])

    # Prepare final clean DataFrame
    processed = pd.DataFrame({
        "conversation_id": merged["tweet_id_customer"].astype(str),
        "customer_text": merged["text_customer"].astype(str).str.strip(),
        "brand_reply_text": merged["text_brand"].astype(str).str.strip(),
        "created_at": merged["created_at_customer"].astype(str),
        "brand": brand
    })

    # Drop empty or extremely short customer messages (< 5 chars)
    processed = processed[processed["customer_text"].str.len() >= 5].copy()

    # Generate cleaned version for vectorization
    processed["customer_text_clean"] = processed["customer_text"].apply(clean_text_for_modeling)

    # Save to disk
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    processed.to_csv(OUTPUT_CONVERSATIONS_PATH, index=False)
    print(f"[preprocessing] Saved {len(processed):,} processed conversations to {OUTPUT_CONVERSATIONS_PATH}")

    return processed


if __name__ == "__main__":
    load_and_preprocess()
