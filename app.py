# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # pymupdf
import docx2txt
import io
import re
import os

# NLP & embeddings
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import sentence-transformers; if not available we'll use TF-IDF fallback
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SBERT = True
except Exception as e:
    HAS_SBERT = False

# Skill list (put in a separate file if you prefer)
TECH_SKILLS = [
    "python","java","javascript","typescript","c++","c","sql","pytorch","tensorflow",
    "scikit-learn","keras","xgboost","lightgbm","nlp","bert","cnn","lstm","transformers",
    "docker","kubernetes","jenkins","ci/cd","github actions","aws","azure","gcp","ec2","s3",
    "lambda","postgresql","mysql","mongodb","redis","django","flask","fastapi","react","redux",
    "html","css","git","postman","linux","mlflow","airflow","prefect","kafka","onnx","sagemaker"
]

nlp = spacy.load("en_core_web_trf")  # better NER than sm

# initialize sbert model lazily
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
sbert = None
def get_sbert():
    global sbert
    if not HAS_SBERT:
        return None
    if sbert is None:
        sbert = SentenceTransformer(SBERT_MODEL_NAME)
    return sbert

app = Flask(__name__)
CORS(app)

# ---------- File extraction ----------
def extract_text_from_file(file_storage):
    filename = file_storage.filename or ""
    ext = filename.split(".")[-1].lower()
    stream = file_storage.stream
    stream.seek(0)

    # PDF via PyMuPDF
    if ext == "pdf":
        try:
            data = stream.read()
            doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc)
            if text and text.strip():
                return text
        except Exception:
            pass

        # fallback: treat as bytes -> attempt decode
        try:
            stream.seek(0)
            return stream.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # DOCX
    if ext == "docx":
        try:
            # docx2txt can accept a temporary file path; we write bytes to temp
            tmp_path = f"/tmp/{os.path.basename(filename)}"
            with open(tmp_path, "wb") as f:
                f.write(stream.read())
            text = docx2txt.process(tmp_path)
            try:
                os.remove(tmp_path)
            except:
                pass
            return text or ""
        except Exception:
            stream.seek(0)
            return stream.read().decode("utf-8", errors="ignore")

    # TXT/others
    try:
        stream.seek(0)
        return stream.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

# ---------- Skill extraction ----------
def extract_skills_from_text(text):
    found = set()
    low = text.lower()
    # simple substring matching
    for s in TECH_SKILLS:
        # special-case: match word boundaries for short tokens
        if re.search(r'\b' + re.escape(s) + r'\b', low):
            found.add(s)
    return sorted(list(found))

# ---------- Embedding-based similarity (with fallback) ----------
def compute_similarity_emb(resume_text, jd_text):
    # Try SBERT
    model = get_sbert()
    if model:
        emb_resume = model.encode(resume_text, convert_to_tensor=True)
        emb_jd = model.encode(jd_text, convert_to_tensor=True)
        score = float(util.pytorch_cos_sim(emb_resume, emb_jd).item())
        return score, "sbert"
    # Fallback TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    docs = [resume_text, jd_text]
    X = vectorizer.fit_transform(docs)
    score = cosine_similarity(X[0:1], X[1:2])[0][0]
    return float(score), "tfidf"

# ---------- Gap analysis ----------
def jd_gap_analysis(resume_text, jd_text):
    # extract skills from JD and resume
    jd_skills = extract_skills_from_text(jd_text)
    resume_skills = extract_skills_from_text(resume_text)

    jd_set = set(jd_skills)
    resume_set = set(resume_skills)

    missing = sorted(list(jd_set - resume_set))
    common = sorted(list(jd_set & resume_set))

    coverage = (len(common) / len(jd_set))*100 if len(jd_set)>0 else 0.0

    # Also provide top-N suggested keywords for resume based on jd TF-IDF (optional)
    # simple approach: return tokens in JD not in TECH_SKILLS but frequent (basic)
    return {
        "jd_skills": jd_skills,
        "resume_skills": resume_skills,
        "missing_skills": missing,
        "common_skills": common,
        "coverage_percent": round(coverage,2)
    }

def generate_jd_recommendations(resume_text, jd_text):
    resume_low = resume_text.lower()
    jd_low = jd_text.lower()

    # Extract skills from JD using keywords
    jd_skills = extract_skills_from_text(jd_text)
    resume_skills = extract_skills_from_text(resume_text)

    missing_skills = list(set(jd_skills) - set(resume_skills))

    recommendations = []

    # Missing technical skills
    for skill in missing_skills:
        recommendations.append(
            f"Consider adding '{skill}' experience as the JD requires it but it is missing in your resume."
        )

    # Role responsibilities
    duties = []
    for line in jd_text.split("\n"):
        if any(x in line.lower() for x in ["responsible", "design", "develop", "build", "manage"]):
            duties.append(line.strip())

    for duty in duties:
        if duty.lower()[:20] not in resume_low:
            recommendations.append(
                f"The JD mentions: '{duty}'. Add a relevant bullet point in your experience section."
            )

    # Action verbs check
    action_verbs = ["developed", "built", "designed", "optimized", "automated", "implemented"]
    missing_verbs = [v for v in action_verbs if v not in resume_low]

    if len(missing_verbs) > 3:
        recommendations.append(
            "Add more strong action verbs such as: " + ", ".join(missing_verbs[:4])
        )

    return recommendations

def ats_friendly_recommendations(resume_text):
    suggestions = []

    # Basic structure checks
    if "experience" not in resume_text.lower():
        suggestions.append("Add an EXPERIENCE section — ATS systems expect it.")

    if "skills" not in resume_text.lower():
        suggestions.append("Add a SKILLS section containing your core technical skills.")

    if len(resume_text.split()) < 250:
        suggestions.append("Resume is short — ATS may score it low. Add more detailed bullet points.")

    # Bullet formatting
    if "•" not in resume_text and "-" not in resume_text:
        suggestions.append("Use bullet points — ATS systems parse bullets better than paragraphs.")

    # Contact info check
    if "@" not in resume_text:
        suggestions.append("Include a professional email — ATS requires contact details.")

    # PDF formatting issues
    if "endstream" in resume_text:
        suggestions.append("Your PDF has formatting issues. Re-export as a clean PDF or use a template.")

    return suggestions


def calculate_ats_score(embed_score_pct, coverage_pct, structural_issues_count):
    structure_score = max(0, 100 - structural_issues_count * 10)
    final_score = (
        embed_score_pct * 0.40 +
        coverage_pct * 0.40 +
        structure_score * 0.20
    )
    return round(final_score, 2)


# ---------- Routes ----------
@app.route("/parse", methods=["POST"])
def parse_route():
    if "resume" not in request.files:
        return jsonify({"error":"No resume file uploaded"}), 400
    f = request.files["resume"]
    text = extract_text_from_file(f)
    entities = [(ent.text, ent.label_) for ent in nlp(text).ents][:200]
    skills = extract_skills_from_text(text)
    return jsonify({
        "word_count": len(text.split()),
        "entities": entities,
        "skills_extracted": skills,
        "raw_text_excerpt": text[:2000]
    })

@app.route("/full-ats-report", methods=["POST"])
def full_ats_report():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text","")
    jd_text = data.get("jd_text","")

    # Embedding score
    emb_score, method = compute_similarity_emb(resume_text, jd_text)
    emb_pct = round(emb_score * 100, 2)

    # Gap analysis
    gap = jd_gap_analysis(resume_text, jd_text)

    # JD-based improvement
    jd_rec = generate_jd_recommendations(resume_text, jd_text)

    # Structural ATS suggestions
    ats_rec = ats_friendly_recommendations(resume_text)

    # ATS Score
    ats_score = calculate_ats_score(
        embed_score_pct=emb_pct,
        coverage_pct=gap["coverage_percent"],
        structural_issues_count=len(ats_rec)
    )

    return jsonify({
        "embedding_score_pct": emb_pct,
        "jd_coverage_pct": gap["coverage_percent"],
        "method": method,
        "missing_skills": gap["missing_skills"],
        "common_skills": gap["common_skills"],
        "jd_recommendations": jd_rec,
        "ats_friendly_suggestions": ats_rec,
        "final_ats_score": ats_score
    })


@app.route("/embed-score", methods=["POST"])
def embed_score_route():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text","")
    jd_text = data.get("jd_text","")
    if not (resume_text and jd_text):
        return jsonify({"error":"Please send resume_text and jd_text"}), 400

    score, method = compute_similarity_emb(resume_text, jd_text)
    # convert to percentage 0-100
    pct = round(float(score)*100,2)
    return jsonify({"method":method,"similarity_score":score,"similarity_pct":pct})

@app.route("/gap-analysis", methods=["POST"])
def gap_route():
    data = request.get_json(force=True)
    resume_text = data.get("resume_text","")
    jd_text = data.get("jd_text","")
    if not (resume_text and jd_text):
        return jsonify({"error":"Please send resume_text and jd_text"}), 400

    gap = jd_gap_analysis(resume_text, jd_text)
    return jsonify(gap)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
