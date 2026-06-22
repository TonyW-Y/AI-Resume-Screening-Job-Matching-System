import torch
import chromadb
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from data_prep import clean_text, truncate_to_words

# Paths
CHROMA_PATH = "chromadb_store"


# set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# load ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    "resumes",
    metadata={"hnsw:space": "cosine"}
)

# Get all resumes From DB
def get_all_resumes(category_filter=None):
    results = collection.get()
    resumes = []
    # loop through results and build a list
    for i in range(len(results["ids"])):
        resumes.append({
            "id": results["ids"][i],
            "category": results["metadatas"][i]["category"],
            "resume_preview": results["documents"][i][:200]
        })
    # filter by category if provided
    if category_filter:
        resumes = [r for r in resumes if r["category"] == category_filter]
    return resumes

# Delete resume
def delete_resume(resume_id: str):
    collection.delete(ids=[resume_id])
    return f"Deleted Resume {resume_id}"

def add_resume_from_text(text: str, category: str):
    # text prep
    result = truncate_to_words(clean_text(text))
    
    # Get next available ID
    existing_ids = collection.get()["ids"]
    next_id = str(max([int(id) for id in existing_ids]) + 1) if existing_ids else "0"

    # Embed the text
    embedding = embed_model.encode(result).tolist()
    
    # Add to ChromaDB
    collection.add(
        embeddings=[embedding],
        documents=[result],
        metadatas=[{"category": category}],
        ids=[next_id]
    )
    
    return f"✅ Resume added with ID {next_id} under category {category}"


def add_resume_from_pdf(pdf_path: str, category: str):
    # open pdf
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return add_resume_from_text(text, category)
        



