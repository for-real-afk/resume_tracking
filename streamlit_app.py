# streamlit_app.py
import streamlit as st
import requests
import io

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="ATS Resume Analyzer + Embeddings", layout="wide")
st.title("ATS Resume Analyzer — Embeddings & JD Gap Analysis")

col1, col2 = st.columns([2,1])

with col1:
    resume_file = st.file_uploader("Upload resume (pdf, docx, txt)", type=["pdf","docx","txt"])
    jd_text = st.text_area("Paste job description (JD) here", height=250)
    st.info("You can also test quickly with the uploaded resume file path: `/mnt/data/Deepanshu-Sonwane-FlowCV-Resume-20251013 (3).pdf`")

with col2:
    if st.button("Parse resume"):
        if not resume_file:
            st.error("Upload a resume first.")
        else:
            files = {"resume": (resume_file.name, resume_file.getvalue())}
            r = requests.post(f"{API_URL}/parse", files=files)
            if r.ok:
                data = r.json()
                st.subheader("Parsed Resume")
                st.json(data)
            else:
                st.error(r.text)

    if st.button("Compute embedding similarity (resume ↔ JD)"):
        if not resume_file or not jd_text.strip():
            st.error("Upload resume and paste JD.")
        else:
            # convert file -> text locally before calling /embed-score
            # but the backend expects resume_text, so extract text here or call parse endpoint
            files = {"resume": (resume_file.name, resume_file.getvalue())}
            r = requests.post(f"{API_URL}/parse", files=files)
            if not r.ok:
                st.error("Could not parse resume.")
            else:
                resume_text = r.json().get("raw_text_excerpt","")
                payload = {"resume_text": resume_text, "jd_text": jd_text}
                r2 = requests.post(f"{API_URL}/embed-score", json=payload)
                if r2.ok:
                    out = r2.json()
                    st.metric("Matching Score (%)", f"{out['similarity_pct']}%")
                    st.write("Method used:", out["method"])
                else:
                    st.error(r2.text)

    if st.button("JD Gap Analysis (skills coverage)"):
        if not resume_file or not jd_text.strip():
            st.error("Upload resume and paste JD.")
        else:
            files = {"resume": (resume_file.name, resume_file.getvalue())}
            r = requests.post(f"{API_URL}/parse", files=files)
            if not r.ok:
                st.error("Could not parse resume.")
            else:
                resume_text = r.json().get("raw_text_excerpt","")
                payload = {"resume_text": resume_text, "jd_text": jd_text}
                r2 = requests.post(f"{API_URL}/gap-analysis", json=payload)
                if r2.ok:
                    out = r2.json()
                    st.subheader("JD Gap Analysis")
                    st.write(f"Coverage: {out['coverage_percent']}%")
                    st.write("Common skills:", out["common_skills"])
                    st.write("Missing skills (from JD):", out["missing_skills"])
                else:
                    st.error(r2.text)
