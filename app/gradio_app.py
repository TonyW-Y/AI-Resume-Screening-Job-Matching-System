import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rank_resumes import search_resumes

def rank_resumes_ui(jd_text: str, top_k: int):
    result = search_resumes(jd_text, top_k)

    # Build candidates string
    candidates_str = f"Predicted Category: {result['predicted_category']}\n\n"
    for candidate in result["ranked_candidates"]:
        candidates_str += f"Rank {candidate['rank']}: {candidate['category']}\n"
        candidates_str += f"Final Score: {candidate['final_score']} | Similarity: {candidate['similarity_score']}\n"
        candidates_str += f"Preview: {candidate['resume_preview']}\n\n"
    
    return candidates_str, result["llm_report"]

with gr.Blocks(title="AI Resume Screener") as demo:
    gr.Markdown("# 🤖 AI Resume Screener")
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

if __name__ == "__main__":
    demo.launch()