# 🧠 ATS Resume Analyzer — Enterprise Edition (v2.0.0)

An enterprise-grade ATS (Applicant Tracking System) resume intelligence engine that analyzes resumes, extracts skills, compares them against a job description using semantic AI, highlights gaps, generates tailored recommendations, and outputs a professional ATS report (PDF included).

## 🚀 Key Features

### 🔍 1. Resume Parsing Engine
- Extracts text from:
  - **PDF** (PyMuPDF + pdfplumber fallback)
  - **DOCX** (docx2txt)
  - **TXT**
- Cleans formatting and removes PDF artifacts.

### 🤖 2. AI Skill Extraction (Hybrid Engine)
Uses three layers:
1. **spaCy Transformer NER**
2. **SkillNER skill ontology**
3. **Keyword dictionary of 200+ tech skills**

### 🧬 3. Enterprise JD Gap Analysis
- Extracts JD skills + duties  
- Detects missing skills in resume  
- Computes JD Coverage %  
- Duty-level alignment  

### 🧠 4. Semantic Matching via MPNet + FAISS
- **Sentence-transformers (all-mpnet-base-v2)** for embeddings  
- **FAISS** for fast semantic search across resume sentences  
- Outputs semantic alignment score  

### 📊 5. Enterprise ATS Score v2
Weights:
- 35% Embedding similarity  
- 35% JD skill coverage  
- 15% Resume structure quality  
- 15% Semantic alignment via FAISS  

### ✍ 6. Tailored Resume Suggestions
- Auto-generated **action bullets** based on JD  
- Missing skill suggestions  
- ATS optimization recommendations  

### 📄 7. Exportable PDF Report
Professional ATS report with:
- ATS Score  
- Missing vs common skills  
- Tailored bullets  
- Recommendations  

### 🖥 8. Streamlit Enterprise UI
- Upload resume  
- Paste JD  
- Generate ATS Report  
- Download PDF  

---

## 🏗 Architecture

