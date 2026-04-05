# ─── SachBol AI — Dockerfile (local development, full ML stack) ───
# Build:  docker build -t sachbol .
# Run:    docker run --env-file .env -p 5000:5000 sachbol

FROM python:3.11-slim

# System deps for tokenizers / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cached separately from code)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download NLTK corpora for TextBlob
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

EXPOSE 5000

# Run with gunicorn — use_reloader=False is set inside app.py
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", \
     "--timeout", "180", \
     "backend.app:app"]
