import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:5000"   # your Flask backend

st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")

st.title("📄⚡ ATS Resume Analyzer with JD Matching")
st.write("Upload your resume and compare it with a job description.")

# --- File uploader ---
resume_file = st.file_uploader("Upload Resume (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

# --- Job Description ---
jd_text = st.text_area("Paste Job Description Here", height=200)

col1, col2 = st.columns(2)

# ---------------------- PARSE RESUME ----------------------
if col1.button("📘 Parse Resume"):
    if not resume_file:
        st.error("Please upload a resume file.")
    else:
        files = {"resume": (resume_file.name, resume_file.getvalue(), resume_file.type)}
        response = requests.post(f"{API_URL}/parse", files=files)
        
        if response.status_code == 200:
            data = response.json()
            st.subheader("🔍 Extracted Resume Details")
            st.json(data["parsed_resume"])
        else:
            st.error("Error parsing the resume. Backend not responding.")

# ---------------------- ATS SCORE ----------------------
if col2.button("📊 Calculate ATS Score"):
    if not resume_file or not jd_text.strip():
        st.error("Please upload a resume and paste JD.")
    else:
        # Convert file → text for ATS
        resume_bytes = resume_file.getvalue()
        resume_text = resume_bytes.decode("utf-8", errors="ignore")

        payload = {
            "resume_text": resume_text,
            "jd_text": jd_text
        }

        response = requests.post(f"{API_URL}/ats-score", json=payload)

        if response.status_code == 200:
            score = response.json()["ats_score"]
            st.subheader("🎯 ATS Score")
            st.metric("Matching Score", f"{score}%")
        else:
            st.error("Error calculating ATS score.")

# ---------------------- IMPROVEMENTS ----------------------
if st.button("💡 Get Improvement Suggestions"):
    if not resume_file or not jd_text.strip():
        st.error("Please upload a resume and paste JD.")
    else:
        resume_bytes = resume_file.getvalue()
        resume_text = resume_bytes.decode("utf-8", errors="ignore")

        response = requests.post(f"{API_URL}/improvements", json={
            "resume_text": resume_text,
            "jd_text": jd_text
        })

        if response.status_code == 200:
            suggestions = response.json()["tips"]
            st.subheader("💡 Recommended Improvements")
            for tip in suggestions:
                st.write(f"👉 {tip}")
        else:
            st.error("Backend error while generating improvements.")
