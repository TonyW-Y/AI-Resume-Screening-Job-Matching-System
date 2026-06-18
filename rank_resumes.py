import torch
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from ffn import ffn
from typing import Dict, Any
import joblib
from build_db import build_chromadb


# Paths
OLLAMA_MODEL = "llama3.2:3b"
CHROMA_PATH = "chromadb_store"
EMBEDDING_FILE = "embeddings/resumes.npy"
CSV_FILE = "data/master_resumes.csv"
CLASSIFIER_MODEL = "models/classifier_model.pth"


# set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load embedding model
embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Load FFN classifier
classifier = ffn().to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location=device))
classifier.eval()

# load ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)  # creates/opens the DB
collection = client.get_or_create_collection(
    "resumes", 
    metadata={"hnsw:space": "cosine"}
)  # creats/opens collection (like s table in SQL)

# Auto-build DB if empty
if collection.count() == 0:
    build_chromadb()
# loading the encoder
le = joblib.load("models/label_encoder.pkl")


def search_resumes(jd_text: str, top_k: int = 5) -> Dict[str, Any]:
    try:
        jd_embedding = embed_model.encode(jd_text, show_progress_bar=False).reshape(1,-1)

        # run the JD embedding through the classifier
        with torch.no_grad():
            input_tensor = torch.tensor(jd_embedding, dtype=torch.float32).to(device)
            output = classifier(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            predicted_category = le.inverse_transform([predicted_class])[0]


        results = collection.query(
        query_embeddings=jd_embedding.tolist(),
        n_results=top_k
        )

        # Guard against empty results
        if not results["documents"] or len(results["documents"][0]) == 0:
            return {
                "error": "No results found in ChromaDB. Run build_chromadb() first.",
                "predicted_category": None,
                "ranked_candidates": [],
                "llm_report": None
            }

        # scoring
        ranked = []
        for i in range(len(results["documents"][0])):
            category = results["metadatas"][0][i]["category"]
            similarity_score = 1 - results["distances"][0][i]  # ChromaDB returns distances not similarities
            category_match = 1.0 if category == predicted_category else 0.0
            final_score = (0.3 * category_match) + (0.7 * similarity_score)
            ranked.append({
                "rank": i + 1,
                "category": category,
                "similarity_score": round(similarity_score, 4),
                "category_match": category_match,
                "final_score": round(final_score, 4),
                "resume_preview": results["documents"][0][i][:200]
            })

        # Sort by final score
        ranked = sorted(ranked, key=lambda x: x["final_score"], reverse=True)

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
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
            )

        report = response["message"]["content"]
        return {
                "predicted_category": predicted_category,
                "ranked_candidates": ranked,
                "llm_report": report
            }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "error": str(e),
            "predicted_category": None,
            "ranked_candidates": [],
            "llm_report": None
        }
    
def main():
    result = search_resumes("Looking for a software engineer with Python and machine learning experience")
    
    print(f"Predicted Category: {result['predicted_category']}")

    print("\n── Ranked Candidates ──")
    for candidate in result["ranked_candidates"]:
        print(f"Rank {candidate['rank']}: {candidate['category']} | Final Score: {candidate['final_score']} | Similarity: {candidate['similarity_score']}")
    
    print("\n── LLM Hiring Report ──")
    print(result["llm_report"])


if __name__ == "__main__":
    main()