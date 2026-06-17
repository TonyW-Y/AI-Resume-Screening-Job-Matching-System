import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = "embeddings/resumes.npy"
CSV_FILE = "data/master_resumes.csv"
MODEL_NAME = "all-mpnet-base-v2"

device = "mps" if torch.backends.mps.is_available() else "cpu"

def search_resumes(jd_text: str, top_k: int = 5):
    # load files
    embeddings = np.load(EMBEDDING_FILE)
    df = pd.read_csv(CSV_FILE)

    # initialize Model and turn job description into vector(list of numbers)
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    jd_embedding = model.encode(jd_text, show_progress_bar=True).reshape(1,-1)

    # Calculate cosine similarity
    scores = cosine_similarity(jd_embedding, embeddings)

    # get top 5 most similar resumes
    top_indices = np.argsort(scores[0])[::-1][:top_k]
    results = []
    for i in top_indices:
        results.append({
            "rank": len(results) + 1,
            "category": df.iloc[i]["category"],
            "similarity_score": round(float(scores[0][i]), 4),
            "resume_preview": df.iloc[i]["resume_text"][:200]
        })
    return results

def main():
    results = search_resumes("Looking for a software engineer with Python and machine learning experience")
    for r in results:
        print(r)
if __name__ == "__main__":
    main()
