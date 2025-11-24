# app.py (enterprise-level)
import os
import io
import re
import json
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fitz  # PyMuPDF
import docx2txt
import numpy as np
import faiss
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer, util
from reportlab.pdfgen import canvas
from datetime import datetime

# optional SkillNER
try:
    from skillNer.skill_extractor_class import SkillExtractor
    from skillNer.general_params import SKILL_DB
    HAS_SKILLNER = True
except Exception:
    HAS_SKILLNER = False

# ------ Config ------
SBERT_MODEL = "all-mpnet-base-v2"   # higher-quality embeddings
VECTOR_DIM = 768                    # mpnet dim
EMBED_DB_PATH = "embed_index.faiss"
EMBED_META_PATH = "embed_meta.json"

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_trf")  # transformer-based NER
sbert = SentenceTransformer(SBERT_MODEL)

if HAS_SKILLNER:
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher=None)

# Tech skill dictionary (extendable)
TECH_SKILLS = [
    "python","java","javascript","typescript","c++","c","sql","pytorch","tensorflow",
    "scikit-learn","keras","xgboost","lightgbm","nlp","bert","cnn","lstm","transformers",
    "docker","kubernetes","jenkins","ci/cd","github actions","aws","azure","gcp","ec2","s3",
    "lambda","postgresql","mysql","mongodb","redis","django","flask","fastapi","react","redux",
    "html","css","git","postman","linux","mlflow","airflow","prefect","kafka","onnx","sagemaker"
]

# ------- Utilities: file extraction -------
def extract_text_from_file(file_storage):
    filename = file_storage.filename or "file"
    ext = filename.split(".")[-1].lower()
    stream = file_storage.stream
    stream.seek(0)
    if ext == "pdf":
        try:
            data = stream.read()
            doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
            pages = [p.get_text("text") for p in doc]
            text = "\n".join(pages)
            if text.strip():
                return text
        except Exception:
            pass
        # fallback pdfplumber
        try:
            stream.seek(0)
            with pdfplumber.open(io.BytesIO(stream.read())) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            return text
        except Exception:
            stream.seek(0)
            return stream.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(stream.read())
        tmp.close()
        text = docx2txt.process(tmp.name)
        try:
            os.unlink(tmp.name)
        except:
            pass
        return text or ""
    else:
        stream.seek(0)
        return stream.read().decode("utf-8", errors="ignore")

# ------- Skill extraction -------
def extract_skills(text):
    text_low = text.lower()
    skills = set()

    # 1. SkillNER (if available)
    if HAS_SKILLNER:
        try:
            out = skill_extractor.annotate(text)
            for r in out.get("results", []):
                skills.add(r.get("text","").lower())
        except Exception:
            pass

    # 2. Keyword match
    for s in TECH_SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text_low):
            skills.add(s)

    # 3. spaCy NER heuristic: collect ORG/PRODUCT/TECH-like tokens (optional)
    doc = nlp(text[:4000])  # limit length for speed
    for ent in doc.ents:
        if ent.label_ in {"ORG","PRODUCT","WORK_OF_ART","TECH"}:
            skills.add(ent.text.lower())

    return sorted([s for s in skills if len(s) > 1])

# ------- Sentence splitting for embedding index -------
def split_sentences(text, min_len=20):
    # naive split using sentences
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents if len(sent.text.strip())>=min_len]
    # fallback chunking
    if not sents:
        chunks = []
        words = text.split()
        chunk_size = 50
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
        return chunks
    return sents

# ------- Build or load FAISS index -------
def build_faiss_index(text, force_rebuild=False):
    # create vectors for sentences and persist
    sents = split_sentences(text)
    if not sents:
        return None
    embeddings = sbert.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    # save meta (sentences)
    meta = {"sentences": sents}
    faiss.write_index(index, EMBED_DB_PATH)
    with open(EMBED_META_PATH, "w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"index": index, "sentences": sents}

def load_faiss_index():
    if not os.path.exists(EMBED_DB_PATH) or not os.path.exists(EMBED_META_PATH):
        return None
    index = faiss.read_index(EMBED_DB_PATH)
    with open(EMBED_META_PATH, "r", encoding="utf8") as f:
        meta = json.load(f)
    return {"index": index, "sentences": meta.get("sentences", [])}

# ------- semantic similarity -------
def semantic_similarity(resume_text, jd_text):
    # encode both and compute cosine similarity
    emb_resume = sbert.encode(resume_text, convert_to_tensor=True, show_progress_bar=False)
    emb_jd = sbert.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)
    score = util.pytorch_cos_sim(emb_resume, emb_jd).item()
    return float(score)

# ------- JD gap analysis -------
def jd_gap_analysis(resume_text, jd_text):
    jd_sk = extract_skills(jd_text)
    res_sk = extract_skills(resume_text)
    jd_set = set(jd_sk); res_set = set(res_sk)
    missing = sorted(list(jd_set - res_set))
    common = sorted(list(jd_set & res_set))
    coverage = round((len(common)/len(jd_set))*100,2) if jd_set else 0.0
    return {"jd_skills": jd_sk, "resume_skills": res_sk, "missing_skills": missing, "common_skills": common, "coverage_percent": coverage}

# ------- ATS structure checks -------
def ats_structure_checks(text):
    suggestions = []
    t = text.lower()
    # basic structure
    if "experience" not in t:
        suggestions.append("Add an EXPERIENCE section.")
    if "skills" not in t:
        suggestions.append("Add a SKILLS section containing core tech skills.")
    if "contact" not in t and "@" not in t:
        suggestions.append("Include email/phone in header.")
    # bullet usage
    if "•" not in text and "-" not in text:
        suggestions.append("Use bullet points for achievements.")
    # PDF artifact check
    if "endstream" in text or "obj" in text:
        suggestions.append("PDF contains artifacts; re-export as a clean PDF or use a different template.")
    return suggestions

# ------- Bullet rewriter (template-based) -------
def generate_action_bullets(jd_text, missing_skills, resume_text, top_n=5):
    # derive duty phrases from JD and generate templated bullets
    bullets = []
    lines = [l.strip() for l in jd_text.splitlines() if l.strip()]
    duty_lines = [l for l in lines if any(k in l.lower() for k in ["develop","build","design","deploy","tune","optimize","work with","collaborate","implement"])]
    # take up to top_n duties
    for i, duty in enumerate(duty_lines[:top_n]):
        # generate templated bullet
        skill = missing_skills[i] if i < len(missing_skills) else ""
        bullet = f"• { 'Led' if 'lead' in duty.lower() else 'Implemented' } {duty.split('.')[0]} using {skill if skill else 'relevant tools and frameworks'}, delivering measurable improvements."
        bullets.append(bullet)
    # if no duties derived, fallback to suggestions from missing skills
    for s in missing_skills:
        bullets.append(f"• Worked on projects using {s} to build and fine-tune models, improving model performance and inference latency.")
    return bullets[:top_n]

# ------- ATS score computation (enterprise) -------
def compute_ats_score_v2(emb_pct, coverage_pct, structure_penalties, semantic_alignment_pct):
    # weights tuned for enterprise:
    # Embedding alignment 35%, JD coverage 35%, Structure 15%, Semantic alignment 15%
    struct_score = max(0, 100 - structure_penalties * 15)  # each penalty 15 points
    score = emb_pct*0.35 + coverage_pct*0.35 + struct_score*0.15 + semantic_alignment_pct*0.15
    return round(score,2)

# ------- PDF report generation -------
def generate_pdf_report(report_json, out_path):
    c = canvas.Canvas(out_path)
    c.setFont("Helvetica", 12)
    y = 800
    c.drawString(40, y, f"ATS Enterprise Report - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    y -= 30
    # Basic metrics
    c.drawString(40, y, f"Final ATS Score: {report_json.get('final_ats_score')}")
    y -= 20
    c.drawString(40, y, f"Embedding Score (%): {report_json.get('embedding_score_pct')}")
    y -= 20
    c.drawString(40, y, f"JD Coverage (%): {report_json.get('jd_coverage_pct')}")
    y -= 30
    c.drawString(40, y, "Missing skills:")
    y -= 20
    for s in report_json.get("missing_skills", []):
        c.drawString(60, y, f"- {s}")
        y -= 15
    y -= 10
    c.drawString(40, y, "Top tailoring suggestions:")
    y -= 20
    for r in report_json.get("jd_recommendations", [])[:10]:
        c.drawString(60, y, f"- {r[:110]}")
        y -= 15
    c.save()

# ------- Routes -------
@app.route("/parse", methods=["POST"])
def parse_route():
    if "resume" not in request.files:
        return jsonify({"error":"No resume uploaded"}), 400
    f = request.files["resume"]
    text = extract_text_from_file(f)
    entities = [(ent.text, ent.label_) for ent in nlp(text).ents][:300]
    skills = extract_skills(text)
    return jsonify({"word_count": len(text.split()), "entities": entities, "skills_extracted": skills, "raw_text": text})

@app.route("/enterprise-report", methods=["POST"])
def enterprise_report():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text", "")
    jd_text = data.get("jd_text", "")
    if not (resume_text and jd_text):
        return jsonify({"error":"resume_text & jd_text required"}), 400

    # Embedding similarity
    emb_score = semantic_similarity(resume_text, jd_text)
    emb_pct = round(emb_score*100,2)

    # JD gap
    gap = jd_gap_analysis(resume_text, jd_text)
    coverage = gap["coverage_percent"]

    # build FAISS index from resume (for semantic alignment checks)
    try:
        build_faiss_index(resume_text)
        idx_data = load_faiss_index()
        # run a semantic search: top-k sentences matching JD
        jd_emb = sbert.encode([jd_text], convert_to_numpy=True)
        faiss.normalize_L2(jd_emb)
        D, I = idx_data["index"].search(jd_emb, 5)
        # compute semantic alignment percent as how many top sentences include JD skills
        top_sents = [idx_data["sentences"][int(i)] for i in I[0] if int(i) < len(idx_data["sentences"])]
        matched_count = sum(1 for s in top_sents if any(k in s.lower() for k in gap["common_skills"]))
        semantic_alignment_pct = round((matched_count/len(top_sents))*100,2) if top_sents else 0.0
    except Exception:
        semantic_alignment_pct = 0.0

    # structure checks
    struct_recs = ats_structure_checks(resume_text)
    structure_penalties = len(struct_recs)

    # generate JD-driven recommendations and bullets
    jd_recs = []
    # missing skills recs
    for s in gap["missing_skills"]:
        jd_recs.append(f"Consider adding '{s}' experience; include project bullets that mention {s}.")
    # duties recs
    for line in jd_text.splitlines():
        if any(w in line.lower() for w in ["develop","design","deploy","build","collaborate","model"]):
            if line.strip() and line.strip()[:20].lower() not in resume_text.lower():
                jd_recs.append(f"Add evidence for: '{line.strip()}' (show project / bullet with metrics).")

    bullets = generate_action_bullets(jd_text, gap["missing_skills"], resume_text)

    final_score = compute_ats_score_v2(emb_pct, coverage, structure_penalties, semantic_alignment_pct)

    report = {
        "embedding_score_pct": emb_pct,
        "jd_coverage_pct": coverage,
        "semantic_alignment_pct": semantic_alignment_pct,
        "ats_friendly_suggestions": struct_recs,
        "missing_skills": gap["missing_skills"],
        "common_skills": gap["common_skills"],
        "jd_recommendations": jd_recs,
        "tailored_bullets": bullets,
        "final_ats_score": final_score,
        "method": "all-mpnet-base-v2"
    }

    # optional: produce PDF and return path
    out_pdf = f"/tmp/ats_report_{int(datetime.utcnow().timestamp())}.pdf"
    generate_pdf_report(report, out_pdf)
    report["pdf_report_path"] = out_pdf

    return jsonify(report)

# simple health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
