import ollama
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

OLLAMA_MODEL = "llama3.2:3b"
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
    
    # Build context from top 5 resumes
    context = ""
    for i in range(len(results["documents"][0])):
        category = results["metadatas"][0][i]["category"]
        resume_text = results["documents"][0][i][:500]
        context += f"\nCandidate {i+1} ({category}):\n{resume_text}\n"
    
    
    prompt = f"""You are an expert hiring manager and recruiter.

    Job Description:
    {jd_text}

    Here are the top {top_k} retrieved candidates:
    {context}

    Please provide:
    1. A ranked list of candidates from most to least suitable
    2. For each candidate explain WHY they are a good or poor fit
    3. A final hiring recommendation with reasoning

    Be specific and reference actual skills and experience from the resumes."""

    # Ollama call
    response = ollama.chat(
    model=OLLAMA_MODEL,
    messages=[{"role": "user", "content": prompt}]
    )

    report = response["message"]["content"]
    return {"results": results, "report": report}


def main():
    if collection.count() == 0:
        print("⚠️ ChromaDB is empty. Run build_db.py first.")
        return
    result = search_resumes("Looking for a software engineer with Python and machine learning experience")
    print("\n── Top Candidates ──")
    for i in range(len(result["results"]["documents"][0])):
        print(f"Rank {i+1}: {result['results']['metadatas'][0][i]['category']}")
    print("\n── LLM Hiring Report ──")
    print(result["report"])

if __name__ == "__main__":
    main()