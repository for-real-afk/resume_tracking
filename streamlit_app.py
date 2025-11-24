# streamlit_app.py (enterprise UI)
import streamlit as st
import requests
import tempfile
import os

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="ATS Analyzer — Enterprise", layout="wide")
st.title("ATS Resume Analyzer — Enterprise Edition")

left, right = st.columns([2,1])

with left:
    resume_file = st.file_uploader("Upload resume (pdf/docx/txt)", type=["pdf","docx","txt"])
    jd_text = st.text_area("Paste Job Description (JD) here", height=300)
    st.write("Quick test file (already uploaded):")
    st.code("/mnt/data/Deepanshu-Sonwane-FlowCV-Resume-20251013 (3).pdf")

with right:
    if st.button("Parse Resume"):
        if not resume_file:
            st.error("Upload a resume first.")
        else:
            files = {"resume": (resume_file.name, resume_file.getvalue())}
            r = requests.post(f"{API_URL}/parse", files=files)
            if r.ok:
                st.json(r.json())
            else:
                st.error(r.text)

    if st.button("Run Enterprise ATS Report"):
        if not resume_file or not jd_text.strip():
            st.error("Please upload resume and paste JD.")
        else:
            # parse resume locally via backend
            files = {"resume": (resume_file.name, resume_file.getvalue())}
            r = requests.post(f"{API_URL}/parse", files=files)
            if not r.ok:
                st.error("Parsing failed.")
            else:
                resume_text = r.json().get("raw_text","")
                payload = {"resume_text": resume_text, "jd_text": jd_text}
                r2 = requests.post(f"{API_URL}/enterprise-report", json=payload)
                if r2.ok:
                    out = r2.json()
                    st.subheader("Enterprise ATS Report")
                    st.metric("Final ATS Score", out["final_ats_score"])
                    st.write("Embedding Score (%)", out["embedding_score_pct"])
                    st.write("JD Coverage (%)", out["jd_coverage_pct"])
                    st.write("Semantic Alignment (%)", out.get("semantic_alignment_pct"))
                    st.write("Missing Skills:", out["missing_skills"])
                    st.write("Top Tailored Bullets:")
                    for b in out.get("tailored_bullets", []):
                        st.write(b)
                    if out.get("pdf_report_path"):
                        # read and offer download
                        path = out["pdf_report_path"]
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                st.download_button("Download PDF Report", data=f, file_name="ats_report.pdf")
                else:
                    st.error(r2.text)
