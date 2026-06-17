import pandas as pd
import re
import os

RAW_CSV = "data/raw/Resume/Resume.csv"
OUTPUT_CSV = "data/master_resumes.csv"

MAX_WORDS = 250

def clean_text(text: str) -> str:
    # Remove HTML tags (e.g., <br>, <li>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace special whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "")
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\/\+\#]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()

    return text

def truncate_to_words(text: str, max_words: int = MAX_WORDS) -> str:
    # Keep only first max_words words.
    text = text.split()
    text = text[:max_words]
    text = " ".join(text)
    return text

def main():
    # Load the CSV file
    df = pd.read_csv(RAW_CSV)
    print(df.shape)

    # apply the clean_text and truncate_to_words to text
    df["resume_text"] = df["Resume_str"].apply(clean_text)
    df["resume_text"] = df["resume_text"].apply(truncate_to_words)
    print(df["resume_text"].head(2))

    # remove duplicate resumes
    df = df.drop_duplicates(subset="resume_text")
    print(df.shape)

    # rename columns
    df = df[["Category", "resume_text"]].rename(columns={"Category": "category"})
    print(df.columns.tolist())

    # save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()