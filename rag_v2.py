import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "chromadb_store"
EMBEDDING_FILE = "embeddings/resumes.npy"
CSV_FILE = "data/master_resumes.csv"

# set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load embedding Model
embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Open db
client = chromadb.PersistentClient(path=CHROMA_PATH)
# Opens Collection
collection = client.get_or_create_collection(
    "resumes",
    metadata={"hnsw:space": "cosine"}
)

def search_resumes(jd_text: str, top_k: int = 5):
    # Turn job description into vector(list of numbers)
    jd_embedding = embed_model.encode(jd_text, show_progress_bar=True).reshape(1,-1)

    results = collection.query(
    query_embeddings=jd_embedding.tolist(),
    n_results=top_k
    )
    return results

def main():
    if collection.count() == 0:
        print("⚠️ ChromaDB is empty. Run build_db.py first.")
        return
    results = search_resumes("Looking for a software engineer with Python and machine learning experience")
    for i in range(len(results["documents"][0])):
        print(f"Rank {i+1}")
        print(f"Category: {results['metadatas'][0][i]['category']}")
        print(f"Preview: {results['documents'][0][i][:200]}")
        print()

if __name__ == "__main__":
    main()
