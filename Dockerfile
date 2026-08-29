# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# PYTHONUNBUFFERED: logs show up immediately in Render's log stream instead
# of being buffered. PYTHONDONTWRITEBYTECODE: skip .pyc files in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# spaCy's English model is a separate download, not a pip dependency entry —
# processing/spacy_keywords.py loads "en_core_web_sm" by name at runtime, so
# it must exist in the image or that keyword-extraction method fails.
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Informational only — Render ignores EXPOSE and injects the real port via
# the $PORT env var (see CMD below), but it documents intent for anyone
# running the image locally.
EXPOSE 8000

# Shell form (not exec/JSON form) is required here so $PORT is actually
# expanded at container start — Render sets $PORT dynamically and the app
# must bind to it, not to a hardcoded port.
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
