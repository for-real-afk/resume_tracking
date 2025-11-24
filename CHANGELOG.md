✅ Release Notes — v2.0.0 (Enterprise Edition)
🚀 Major Release: ATS Resume Analyzer — Enterprise Edition (v2.0.0)

This release transforms the previously basic ATS resume parser into a full enterprise-grade ATS evaluation system with:

🔥 Major Enhancements

FAISS-powered semantic search for resume sentence similarity.

MPNet (all-mpnet-base-v2) embeddings for state-of-the-art semantic matching.

Hybrid Skill Extraction Engine (spaCy Transformer + SkillNER + Keyword Matching).

JD Gap Analysis Engine:

Missing skills detection

Duty-based gap analysis

Coverage scoring

Semantic alignment scoring

🧠 Intelligent Tailoring Engine

Auto-generates tailored bullet points using JD duties + missing skills.

Recommends resume rewrites for ATS optimization.

📊 Updated ATS Scoring Model (ATS Score v2)

Weighted using:

Embedding similarity (35%)

JD coverage (35%)

ATS structure quality (15%)

Semantic alignment via FAISS (15%)

📄 Exportable PDF ATS Report

Professional ATS report

Includes missing skills, coverage %, embedding %, structural fixes, tailored bullets

🗂 Resume Parsing Enhancements

Supports PDF, DOCX, TXT

Clean extraction using:

PyMuPDF

pdfplumber (fallback)

docx2txt

🖥 Streamlit Enterprise UI

JD + Resume input

ATS Report

PDF Download

Realtime evaluation

🐳 Deployment Support

Added Dockerfile

Added GitHub Actions CI workflow

Added production Gunicorn entrypoint

🆚 Version Bump
Version: 2.0.0
Type: Major Release

📌 Changelog (CHANGELOG.md entry)
v2.0.0 — 2025-11-24
Added

FAISS vector index for semantic search

MPNet SBERT embeddings for high-accuracy matching

New enterprise JD Gap Analysis engine

Hybrid Skill Extraction (spaCy Trf + SkillNER + keyword match)

ATS Score v2 (weighted multi-factor scoring)

PDF ATS report generator

Streamlit Enterprise UI module

Dockerfile for production

GitHub Actions CI workflow

Resume tailoring module (auto bullets & rewrite engine)

Improved

PDF/DOCX/TXT parsing performance

NER accuracy using en_core_web_trf

Stability + modularity of backend architecture

Fixed

Incorrect parsing of PDF binary streams

Missing file path issues in Streamlit