import gradio as gr
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rank_resumes import search_resumes
from resume_manager import get_all_resumes, delete_resume, add_resume_from_text, add_resume_from_pdf

# Categories list
CATEGORIES = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS",
    "AUTOMOBILE", "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT",
    "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA",
    "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
    "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER"
]

def rank_resumes_ui(jd_text: str, top_k: int):
    result = search_resumes(jd_text, top_k)
    candidates_str = f"Predicted Category: {result['predicted_category']}\n\n"
    for candidate in result["ranked_candidates"]:
        candidates_str += f"Rank {candidate['rank']}: {candidate['category']}\n"
        candidates_str += f"Final Score: {candidate['final_score']} | Similarity: {candidate['similarity_score']}\n"
        candidates_str += f"Preview: {candidate['resume_preview']}\n\n"
    return candidates_str, result["llm_report"]

def load_resumes(category_filter):
    filter_val = None if category_filter == "ALL" else category_filter
    resumes = get_all_resumes(filter_val)
    if not resumes:
        return []
    return [[r["id"], r["category"], r["resume_preview"]] for r in resumes]

def delete_resume_ui(resume_id: str):
    if not resume_id.strip():
        return "⚠️ Please enter a resume ID"
    return delete_resume(resume_id.strip())

def add_text_ui(text: str, category: str):
    if not text.strip():
        return "⚠️ Please enter resume text"
    return add_resume_from_text(text, category)

def add_pdf_ui(pdf_file, category: str):
    if pdf_file is None:
        return "⚠️ Please upload a PDF file"
    return add_resume_from_pdf(pdf_file.name, category)

with gr.Blocks(title="AI Resume Screener") as demo:
    gr.Markdown("# 🤖 AI Resume Screener")
    
    with gr.Tabs():
        with gr.Tab("🔍 Search"):
            gr.Markdown("Enter a job description to find the most matching candidates")
            with gr.Row():
                jd_input = gr.Textbox(label="Job Description", lines=5, placeholder="Enter job description here...")
                top_k_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of candidates")
            submit_btn = gr.Button("Find Candidates", variant="primary")
            with gr.Row():
                candidates_output = gr.Textbox(label="Ranked Candidates", lines=15)
                report_output = gr.Textbox(label="LLM Hiring Report", lines=15)
            submit_btn.click(
                fn=rank_resumes_ui,
                inputs=[jd_input, top_k_input],
                outputs=[candidates_output, report_output]
            )

        with gr.Tab("📁 Resume Manager"):
            gr.Markdown("## View, Add, and Delete Resumes")
            
            gr.Markdown("### View Resumes")
            with gr.Row():
                category_filter = gr.Dropdown(
                    choices=["ALL"] + CATEGORIES,
                    value="ALL",
                    label="Filter by Category"
                )
                load_btn = gr.Button("Load Resumes")
            
            resume_table = gr.Dataframe(
                headers=["ID", "Category", "Preview"],
                datatype=["str", "str", "str"],
                label="Resumes"
            )
            load_btn.click(fn=load_resumes, inputs=[category_filter], outputs=[resume_table])

            gr.Markdown("### Delete Resume")
            with gr.Row():
                delete_id_input = gr.Textbox(label="Resume ID to Delete", placeholder="Enter ID e.g. 42")
                delete_btn = gr.Button("Delete", variant="stop")
            delete_output = gr.Textbox(label="Result")
            delete_btn.click(fn=delete_resume_ui, inputs=[delete_id_input], outputs=[delete_output])

            gr.Markdown("### Add Resume from Text")
            with gr.Row():
                add_text_input = gr.Textbox(label="Resume Text", lines=5, placeholder="Paste resume text here...")
                add_category_input = gr.Dropdown(choices=CATEGORIES, label="Category")
            add_text_btn = gr.Button("Add Resume", variant="primary")
            add_text_output = gr.Textbox(label="Result")
            add_text_btn.click(fn=add_text_ui, inputs=[add_text_input, add_category_input], outputs=[add_text_output])

            gr.Markdown("### Add Resume from PDF")
            with gr.Row():
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                pdf_category_input = gr.Dropdown(choices=CATEGORIES, label="Category")
            add_pdf_btn = gr.Button("Upload and Add", variant="primary")
            add_pdf_output = gr.Textbox(label="Result")
            add_pdf_btn.click(fn=add_pdf_ui, inputs=[pdf_upload, pdf_category_input], outputs=[add_pdf_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)