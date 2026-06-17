import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

"""converting text into vectors(list of numbers) so our deeplearning model can use it"""

INPUT_CSV = "data/master_resumes.csv"
OUTPUT_NPY = "embeddings/resumes.npy"

device = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    # load CSV file
    df = pd.read_csv(INPUT_CSV)
    df["resume_text"] = df["resume_text"].fillna("").astype(str)
    print(df.shape)

    # load all-mpnet-base-v2 model (reads text and convers into 768 numbers)
    print(f"Loading model on device: {device}")
    model = SentenceTransformer("all-mpnet-base-v2", device=device)

    # generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(df["resume_text"].tolist(), show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings
    np.save(OUTPUT_NPY, embeddings)
    print(f"✅ Saved to {OUTPUT_NPY}")
    
if __name__ == "__main__":
    main()