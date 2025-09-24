import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
import openai
from openai import AzureOpenAI
import re
from pyresparser import ResumeParser
import tempfile
import shutil

# -------------------------
# CONFIG
# -------------------------
load_dotenv()

# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Free & lightweight model for embeddings

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
    return round(float(score) * 100, 2)  # Convert to percentage

def parse_llm_json(text: str):
    """
    Extract JSON from LLM output, even if it's wrapped in code blocks
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

def process_and_save_resume(uploaded_file, metadata_path="metadata.json"):
    """
    Process the uploaded resume file: 
    - Extract text from the PDF.
    - Parse resume data using PyResParser.
    - Generate a summary using LLM.
    - Combine all data and save to metadata.json.
    """
    # Save the file temporarily and extract its text
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # Step 1: Extract text from the resume PDF
        resume_text = extract_text_from_pdf(file_path)
        
        # Step 2: Parse the resume data using pyresparser
        parsed_data = ResumeParser(file_path).get_extracted_data()

        # Step 3: Generate a professional summary using LLM
        prompt = f"""
        You are a resume expert. Read the following resume text and generate a 2-3 sentence professional summary.
        Resume:
        {resume_text}
        """
        response = openai.ChatCompletion.create(
            model=os.getenv("MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()

        # Step 4: Combine the parsed data with the LLM summary
        parsed_data["summary"] = summary

        # Step 5: Update metadata file with the new entry
        update_metadata_file(parsed_data, uploaded_file.name, metadata_path)

        return parsed_data
    
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def update_metadata_file(new_entry, filename, metadata_path="metadata.json"):
    """
    Add or update resume metadata in the metadata.json file.
    """
    # Load existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = {}
    else:
        metadata = {}

    # Update with new entry (using filename as the key)
    metadata[filename] = new_entry

    # Save back to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

def analyze_with_llm(jd_text, metadata: dict, baseline_score):
    """
    Use LLM to analyze resume metadata against a job description.

    Parameters:
    - jd_text: Job description as plain text.
    - metadata: Parsed resume details from metadata.json.
    - baseline_score: Initial similarity score from embedding comparison.

    Returns:
    - JSON string response from LLM.
    """
    # Create a simplified view of the resume from metadata
    name = metadata.get("name", "Unknown")
    email = metadata.get("email", "Not Provided")
    phone = metadata.get("mobile_number", "Not Provided")
    skills = ", ".join(metadata.get("skills", []))
    experience = metadata.get("experience", "Not Specified")
    education = ", ".join(metadata.get("education", [])) if isinstance(metadata.get("education"), list) else metadata.get("education", "")
    summary = metadata.get("summary", "")

    # Construct a synthetic resume from metadata
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

    # LLM prompt
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
1. First, consider the baseline similarity score: {baseline_score}.
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
        baseline_score = compute_similarity(jd_text, resume_text)
        llm_result_str = analyze_with_llm(jd_text, resume_text, baseline_score)
        llm_result = parse_llm_json(llm_result_str)

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
