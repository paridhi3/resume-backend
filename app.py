import streamlit as st
import fitz  # PyMuPDF
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # free & lightweight

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_text_from_pdf(uploaded_file):
    """Extracts text from PDF using PyMuPDF (fitz)."""
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def compute_similarity(jd_text, resume_text):
    """Compute cosine similarity between JD and Resume embeddings."""
    jd_emb = embedding_model.encode([jd_text])
    resume_emb = embedding_model.encode([resume_text])
    score = cosine_similarity(jd_emb, resume_emb)[0][0]
    return round(float(score) * 100, 2)  # convert to percentage

def analyze_with_llm(jd_text, resume_text, score):
    """Use OpenAI LLM to analyze resume vs JD and return JSON."""
    prompt = f"""
You are an AI-powered Applicant Tracking System (ATS). 
Your task is to analyze how well a given resume matches a job description. 
You must consider **skills, experience, education, and keywords**.

Return output strictly in **valid JSON** with this schema:
{{
  "category": "Suitable / Maybe / Not Suitable",
  "final_score": "0-100 (improved ATS score after reasoning)",
  "reasons": [
    "reason 1",
    "reason 2",
    "reason 3"
  ]
}}

Instructions:
1. First, consider the baseline similarity score: {score}.
2. Adjust the final_score based on skills, experience, and job requirements.
3. Place the resume in one of three categories:
   - Suitable: strong match (generally >70)
   - Maybe: partial match, missing some skills (40–70)
   - Not Suitable: weak match (<40)
4. Provide 3-5 clear, specific reasons for your decision.
5. Do not add any text outside of the JSON format.

Job Description:
{jd_text}

Resume:
{resume_text}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are an expert ATS screening assistant."},
            {"role":"user","content":prompt}
        ]
    )
    raw_output = response.choices[0].message["content"]

    # try parsing JSON safely
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {"category":"Error","final_score":0,"reasons":["Failed to parse LLM output"]}
    return parsed

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📄 AI Resume–JD Analyzer (Prototype)")

jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if jd_file and resume_files:
    jd_text = extract_text_from_pdf(jd_file)
    results = []

    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        baseline_score = compute_similarity(jd_text, resume_text)
        llm_result = analyze_with_llm(jd_text, resume_text, baseline_score)

        results.append({
            "Candidate": resume_file.name,
            "Baseline Score": baseline_score,
            "Final Score": llm_result.get("final_score"),
            "Category": llm_result.get("category"),
            "Reasons": "; ".join(llm_result.get("reasons", []))
        })

    df = pd.DataFrame(results)
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    # Export option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results as CSV", csv, "results.csv", "text/csv")
