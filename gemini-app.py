import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import json

# -------------------------
# CONFIG
# -------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
    """Use Google Gemini LLM to analyze resume vs JD and return JSON."""
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
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    llm_result = response.text
    return llm_result

def parse_gemini_json(text: str):
    """
    Extract JSON from Gemini LLM output, even if it's wrapped in code blocks
    or contains extra whitespace/newlines.
    """
    # Remove code block markers
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()

    # Extract the first {...} block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        return {"category": "Error", "final_score": 0, "reasons": ["Failed to parse LLM output"]}

    # Parse JSON safely
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"category": "Error", "final_score": 0, "reasons": ["Failed to parse LLM output"]}


# -------------------------
# STREAMLIT UI
# -------------------------
st.title("📄 AI Resume Analyzer")

jd_input_type = st.radio("Choose Job Description input method:", ["Upload PDF", "Enter Text"])

jd_text = ""
if jd_input_type == "Upload PDF":
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)
else:
    jd_text = st.text_area("Paste Job Description text here")

resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
results = []    
if jd_text and resume_files:
    for resume_file in resume_files:
        resume_text = extract_text_from_pdf(resume_file)
        baseline_score = compute_similarity(jd_text, resume_text)
        llm_result_str = analyze_with_llm(jd_text, resume_text, baseline_score)
        llm_result = parse_gemini_json(llm_result_str)

        # Flatten reasons into a single string
        reasons_str = " | ".join(llm_result.get("reasons", []))

        results.append({
            "Candidate": resume_file.name,
            "Baseline Score": baseline_score,
            "Final Score": llm_result.get("final_score"),
            "Category": llm_result.get("category"),
            "Reasons": reasons_str
        })

# -------------------------
# Display Results and Export
# -------------------------
if results:
    df = pd.DataFrame(results)
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    # Export CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv,
        file_name="resume_analysis.csv",
        mime="text/csv"
    )

