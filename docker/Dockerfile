
# ============================================================
# PETRA-RAG — gemeinsames Dockerfile für beide Dienste
#
# Zwei Build-Stages in EINER Datei:
#   - Stage "backend":  Ingestion-API (FastAPI, Port 8001)
#   - Stage "frontend": Streamlit-UI  (Port 8501)
#
# docker-compose.yml wählt je Service per "target:" die passende
# Stage aus derselben Datei aus (siehe target: backend / frontend).
# Nach jeder Codeänderung ist ein Rebuild erforderlich:
#   docker compose build ingestion-api
#   docker compose up -d ingestion-api
#
# ============================================================
 
# ---------- Stage: backend ----------------------------------
FROM python:3.11-slim AS backend

WORKDIR /app

# tesseract-ocr + Sprachpakete: OCR gescannter PDFs (FA-ING-05, OCR_LANG=deu+eng)
# poppler-utils: PDF-Hilfswerkzeuge
# curl: für Healthchecks und Diagnose im laufenden Container

RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY data/src /app/src

#  PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch

# ── Python-Abhängigkeiten ──────────────────────────────────────────────────
# Bewusst mit Versionsuntergrenzen statt exakter Pins: reproduzierbar genug
# für die Abgabe, ohne bei jedem Sicherheitsupdate zu brechen.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    chromadb \
    requests \
    python-dotenv \
    PyMuPDF \
    pytesseract \
    Pillow \
    langchain-text-splitters \
    sentence-transformers \
    "httpx>=0.27" \
    "pytest>=8.0" \
    "langchain-community>=0.3" \
    "ddgs>=9.0" \
    "sentence-transformers>=2.7"  \
    "datasets>=2.19" 
    
# ── Modell-Cache ───────────────────────────────────────────────────────────
# HF_HOME zeigt per Compose auf /data/hf_cache (persistentes Bind-Mount).
# Das Cross-Encoder-Modell wird NICHT ins Image gebacken, sondern einmalig
# nach dem ersten Start geladen (MIGRATION.md, Schritt 2):
#
#   docker compose exec ingestion-api python -c \
#     "from sentence_transformers import CrossEncoder; \
#      CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
#
# Grund: Ein Download zur Build-Zeit würde das Image um mehrere hundert MB
# aufblähen und bei jedem Rebuild erneut anfallen. Über das Volume bleibt
# das Modell auch über Rebuilds hinweg erhalten.
# Danach kann TRANSFORMERS_OFFLINE=1 gesetzt werden (NF-02).

WORKDIR /app/src/pipeline

EXPOSE 8001

CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8001"]


# ---------- Stage: frontend -----------------------------------
FROM python:3.11-slim AS frontend

WORKDIR /app
    
RUN pip install --no-cache-dir \
    "streamlit>=1.37,<2.0" \
    "requests>=2.31" \
    "langchain-community>=0.3,<0.4" \
    "ragas==0.2.15" \
    "openpyxl>=3.1"  \
    "streamlit-mic-recorder>=0.0.8" \ 
    "streamlit-webrtc>=0.47.0" \ 
    "faster-whisper>=1.0.01" \
    "av>=11.0.0"   \
    "numpy>=1.24"   \
    "pyttsx3>=2.90" \
    "pandas>=2.0"

RUN apt-get update && apt-get install -y \   
    espeak-ng  \
    ffmpeg

COPY frontend/ /app
    
EXPOSE 8501
    
# server.address wird bewusst per CLI-Flag gesetzt (überschreibt
# .streamlit/config.toml, die dort 127.0.0.1 vorgibt), damit die
# App im Docker-Netzwerk und über den Host-Portmapping erreichbar
# ist. NF-11 (nur localhost) bleibt gewahrt, weil docker-compose.yml
# den Port auf "127.0.0.1:8501:8501" beschränkt.
CMD ["streamlit", "run", "app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]