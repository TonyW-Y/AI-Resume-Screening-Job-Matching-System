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

def build_chromadb():
    # load files
    embedding = np.load(EMBEDDING_FILE)
    df = pd.read_csv(CSV_FILE)

    # replace NaN values to an empty str and forces every value in the column to be a str
    df["resume_text"] = df["resume_text"].fillna("").astype(str)


    client = chromadb.PersistentClient(path=CHROMA_PATH)  # creates/opens the DB
    collection = client.get_or_create_collection(
        "resumes",
        metadata={"hnsw:space": "cosine"}
    )  # creats/opens collection (like s table in SQL)

    # only add if collection is empty
    if collection.count() == 0:
        collection.add(
            embeddings=embedding.tolist(),   # the vectors
            documents=df["resume_text"].tolist(),    # the resume text
            metadatas=[{"category": cat} for cat in df["category"].tolist()],    # extra info like category
            ids=[str(i) for i in range(len(df))]           # unique ID for each resume
        )
        print(f"✅ Added {len(df)} resumes to ChromaDB")
    else:
        print(f"✅ ChromaDB already has {collection.count()} resumes, skipping build")

def search_resumes(jd_text: str, top_k: int = 5):
    # Load the Model and turn job description into vector(list of numbers)
    model = SentenceTransformer("all-mpnet-base-v2", device=device)
    jd_embedding = model.encode(jd_text, show_progress_bar=True).reshape(1,-1)

    client = chromadb.PersistentClient(path=CHROMA_PATH)  # creates/opens the DB
    collection = client.get_or_create_collection("resumes")  # creats/opens collection (like s table in SQL)

    results = collection.query(
    query_embeddings=jd_embedding.tolist(),
    n_results=top_k
    )
    return results

def main():
    build_chromadb()
    results = search_resumes("Looking for a software engineer with Python and machine learning experience")
    
    for i in range(len(results["documents"][0])):
        print(f"Rank {i+1}")
        print(f"Category: {results['metadatas'][0][i]['category']}")
        print(f"Preview: {results['documents'][0][i][:200]}")
        print()

if __name__ == "__main__":
    main()
