import numpy as np
import pandas as pd
import chromadb

CHROMA_PATH = "chromadb_store"
EMBEDDING_FILE = "embeddings/resumes.npy"
CSV_FILE = "data/master_resumes.csv"

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

def main():
    build_chromadb()

if __name__ == "__main__":
    main()
