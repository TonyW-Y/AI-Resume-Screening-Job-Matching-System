from fastapi import FastAPI
from pydantic import BaseModel
from rank_resumes import search_resumes

class RankRequest(BaseModel):
    jd_text: str
    top_k: int = 5


app = FastAPI()

@app.post("/rank")
def rank(request: RankRequest):
    result = search_resumes(request.jd_text, request.top_k)
    return result