import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from rank_resumes import search_resumes
from resume_manager import get_all_resumes, delete_resume, add_resume_from_text

class RankRequest(BaseModel):
    jd_text: str
    top_k: int = 5
    
class AddResumeRequest(BaseModel):
    text: str
    category: str

app = FastAPI()

@app.post("/rank")
def rank(request: RankRequest):
    result = search_resumes(request.jd_text, request.top_k)
    return result


@app.get("/resumes")
def list_resumes(category: str = None):
    return get_all_resumes(category)

@app.delete("/resumes/{resume_id}")
def remove_resume(resume_id: str):
    return delete_resume(resume_id)

@app.post("/resumes")
def add_resume(request: AddResumeRequest):
    return add_resume_from_text(request.text, request.category)
