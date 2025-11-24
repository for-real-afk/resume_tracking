📄 ATS Resume Analyzer — AI-Powered Resume Parsing, Matching & JD Gap Analysis

A full-stack ATS (Applicant Tracking System) analyzer that lets users:

Upload a resume (PDF, DOCX, TXT)

Parse structured information (skills, entities, word count)

Extract technical skills using NLP + keyword models

Compute Resume ↔ JD similarity using

Sentence Transformers Embeddings (MiniLM)

TF-IDF fallback if embeddings unavailable

Perform JD Gap Analysis

Skills missing in the resume

Skills matched

Skill coverage percentage

View everything through a beautiful Streamlit UI

🚀 Tech Stack
Backend

Python

Flask

spaCy (transformer model)

Sentence-Transformers (SBERT)

pdfplumber / PyMuPDF (PDF parsing)

docx2txt

Frontend

Streamlit

Requests

📁 Project Structure
resume-ats-app/
│
├── app.py                     # Flask backend (parsing + embeddings + JD gap analysis)
├── streamlit_app.py           # Streamlit UI
├── skills_list.py             # Defined technical skill dictionary
├── requirements.txt           # Dependencies
│
├── README.md                  # Documentation
└── sample_resumes/            # (Optional) Store test resumes

🛠 Installation
1. Clone the repository
git clone https://github.com/for-real-afk/resume_tracking.git
cd resume-ats-app

2. Create Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate


Mac / Linux

python3 -m venv venv
source venv/bin/activate

3. Install Requirements
pip install -r requirements.txt
python -m spacy download en_core_web_trf


This will also download the SentenceTransformer model on first run.

⚙️ Run the App
Start Backend (Flask)
python app.py


Server runs at:

http://127.0.0.1:5000

Start Frontend (Streamlit)
streamlit run streamlit_app.py


Streamlit UI will open at:

http://localhost:8501

🧠 Key Features
✔ Resume Parsing

Uses spaCy + PyMuPDF/docx2txt to extract:

Clean text

Skills

Entities

Word count

✔ AI-Based Resume ↔ JD Matching

Powered by SBERT (all-MiniLM-L6-v2):

{
  "similarity_pct": 82.6,
  "method": "sbert"
}


Fallback: TF-IDF if embeddings are not available.

✔ JD Gap Analysis

Detects:

Skills found in JD

Skills found in resume

Missing skills

Skill coverage percentage

Example output:

{
  "jd_skills": ["python", "docker", "aws"],
  "resume_skills": ["python", "aws"],
  "missing_skills": ["docker"],
  "coverage_percent": 66.67
}

✔ Clean Streamlit UI

User-friendly interface to:

Upload resume

Paste JD text

View ATS score

View JD gap analysis

View extracted skills

📡 API Endpoints
1. Resume Parsing

POST /parse

Form-Data:

resume: <file>


Returns parsed text, skills, entities, etc.

2. Embedding Score

POST /embed-score

{
  "resume_text": "...",
  "jd_text": "..."
}


Returns:

{ "similarity_pct": 81.2, "method": "sbert" }

3. JD Gap Analysis

POST /gap-analysis

{
  "resume_text": "...",
  "jd_text": "..."
}

📦 requirements.txt

Your requirements.txt should contain:

flask
flask-cors
spacy
pdfplumber
pymupdf
python-docx
docx2txt
sentence-transformers
scikit-learn
nltk
streamlit

📘 Usage Example

Upload resume (PDF/DOCX/TXT)

Paste job description

Click

Parse Resume

Embedding Similarity

JD Gap Analysis

See ATS Score, matched skills, missing keywords.

👨‍💻 Contributing

PRs are welcome!

Suggested improvements:

Add HuggingFace inference API

Support multi-page JD parsing

Add resume tailoring suggestions

Deploy on Render / Railway / HuggingFace Spaces

📄 License

This project is licensed under the MIT License.

⭐ If this project helps you land interviews, give the repo a star! 🌟