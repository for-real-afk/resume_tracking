# Flask-based ATS Resume Parser App
# Structure: app.py main server file

from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import pdfplumber
import docx2txt
import json

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")

# --- Utility functions ---
def extract_text(file_stream, filename):
    ext = filename.split(".")[-1].lower()

    # PDF
    if ext == "pdf":
        with pdfplumber.open(file_stream) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        return text

    # DOCX
    elif ext == "docx":
        text = docx2txt.process(file_stream)
        return text or ""

    # TXT fallback
    else:
        return file_stream.read().decode("utf-8", errors="ignore")

def parse_resume(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["SKILL", "ORG", "WORK_OF_ART"]]
    return {
        "word_count": len(text.split()),
        "entities": [(ent.text, ent.label_) for ent in doc.ents],
        "skills_extracted": list(set(skills))
    }

def match_resume_to_jd(resume_text, jd_text):
    resume_doc = nlp(resume_text)
    jd_doc = nlp(jd_text)
    similarity = resume_doc.similarity(jd_doc)
    return round(float(similarity) * 100, 2)

# --- Routes ---

@app.route("/parse", methods=["POST"])
def parse():
    file = request.files["resume"]
    filename = file.filename
    text = extract_text(file.stream, filename)
    parsed = parse_resume(text)
    return jsonify({"parsed_resume": parsed})

@app.route("/ats-score", methods=["POST"])
def ats_score():
    data = request.json
    resume_text = data["resume_text"]
    jd_text = data["jd_text"]
    score = match_resume_to_jd(resume_text, jd_text)
    return jsonify({"ats_score": score})

@app.route("/improvements", methods=["POST"])
def improvements():
    data = request.json
    resume = data["resume_text"]
    jd = data["jd_text"]

    suggestions = []

    if len(resume.split()) < 250:
        suggestions.append("Resume seems too short. Consider expanding experience bullets.")

    if "team" not in resume.lower():
        suggestions.append("Add teamwork or cross-collaboration experiences.")

    if "project" not in resume.lower():
        suggestions.append("Include more project details with quantified outcomes.")

    # Matching suggestions
    ats = match_resume_to_jd(resume, jd)
    if ats < 60:
        suggestions.append("ATS match is low. Add keywords from JD.")

    return jsonify({"tips": suggestions})


if __name__ == "__main__":
    app.run(debug=True)
