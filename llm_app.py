import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os
from dotenv import load_dotenv
import re
from openai import AzureOpenAI

# -------------------------
# CONFIG
# -------------------------
load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=None)
METADATA_FILE = "metadata.json"

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


def process_and_save_resume(uploaded_file, resume_text):
    """
    Process resume:
    - Extract email + phone with regex
    - Use LLM to extract name, skills, education, experience, summary
    - Save structured data to metadata.json (skip if already exists)
    """
    # 1. Regex for email and phone
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", resume_text)
    phone_match = re.search(r"\+?\d[\d\s\-]{8,}\d", resume_text)
    email = email_match.group(0) if email_match else None
    phone = phone_match.group(0) if phone_match else None

    # 2. Ask LLM for structured parsing
    prompt = f"""
    You are an expert resume parser.
    Extract the following details in strict JSON format:
    {{
      "name": "Candidate Name",
      "skills": ["skill1", "skill2"],
      "education": ["education1", "education2"],
      "experience": ["experience1", "experience2"],
      "summary": "2-3 sentence professional summary"
    }}

    Resume:
    {resume_text}
    """

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    llm_output = response.choices[0].message.content.strip()

    # Parse JSON safely
    try:
        parsed_llm = json.loads(re.sub(r"```json|```", "", llm_output))
    except:
        parsed_llm = {"name": None, "skills": [], "education": [], "experience": [], "summary": llm_output}

    # 3. Merge regex + LLM results
    parsed_data = {
        "file_name": uploaded_file.name,
        "name": parsed_llm.get("name"),
        "email": email,
        "mobile_number": phone,
        "skills": parsed_llm.get("skills", []),
        "education": parsed_llm.get("education", []),
        "experience": parsed_llm.get("experience", []),
        "summary": parsed_llm.get("summary", ""),
        "raw_text": resume_text
    }

    # 4. Save metadata.json (skip duplicates)
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    file_key = parsed_data["file_name"]
    if file_key not in metadata:  # skip duplicate
        metadata[file_key] = parsed_data
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return parsed_data


def compute_similarity(jd_text, resume_text):
    """Compute cosine similarity between JD and Resume embeddings."""
    jd_emb = embedding_model.encode([jd_text])
    resume_emb = embedding_model.encode([resume_text])
    score = cosine_similarity(jd_emb, resume_emb)[0][0]
    return round(float(score) * 100, 2)  # percentage


def parse_llm_json(text: str):
    """Parse JSON safely from LLM output."""
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        return {"category": "Error", "final_score": 0, "reasons": ["Failed to parse LLM output"]}

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"category": "Error", "final_score": 0, "reasons": ["Failed to parse LLM output"]}


def analyze_with_llm(jd_text, metadata: dict, baseline_score):
    """Analyze resume metadata against job description with LLM."""
    name = metadata.get("name", "Unknown")
    email = metadata.get("email", "Not Provided")
    phone = metadata.get("mobile_number", "Not Provided")
    skills = ", ".join(metadata.get("skills", []))
    experience = ", ".join(metadata.get("experience", []))
    education = ", ".join(metadata.get("education", []))
    summary = metadata.get("summary", "")

    resume_synth = f"""
Name: {name}
Email: {email}
Phone: {phone}

Professional Summary:
{summary}

Skills:
{skills}

Experience:
{experience}

Education:
{education}
"""

    prompt = f"""
You are an AI-powered ATS.
Analyze how well a resume matches a job description.
Consider skills, experience, education, and keywords.

Return output strictly in valid JSON:
{{
  "category": "Suitable / Maybe / Not Suitable",
  "final_score": "0-100",
  "reasons": ["reason 1", "reason 2", "reason 3"]
}}

Baseline similarity score: {baseline_score}

Job Description:
{jd_text}

Resume:
{resume_synth}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content

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
        parsed = process_and_save_resume(resume_file, resume_text)
        baseline_score = compute_similarity(jd_text, parsed["raw_text"])
        llm_result_str = analyze_with_llm(jd_text, parsed, baseline_score)
        llm_result = parse_llm_json(llm_result_str)

        if llm_result.get("category") == "Error":
            st.warning(f"⚠️ Failed to parse LLM output for {resume_file.name}: {llm_result['reasons'][0]}")
            continue

        results.append({
            "Candidate": resume_file.name,
            "Baseline Score": baseline_score,
            "Final Score": llm_result.get("final_score"),
            "Category": llm_result.get("category"),
            "Reasons": " | ".join(llm_result.get("reasons", []))
        })

# -------------------------
# Display Results
# -------------------------
if results:
    df = pd.DataFrame(results)
    st.subheader("📊 Results")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results as CSV",
        data=csv,
        file_name="resume_analysis.csv",
        mime="text/csv"
    )
