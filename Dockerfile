# ─── SachBol AI — Gateway Dockerfile (Railway / Render) ───────
# Lightweight proxy only — zero ML dependencies.
# All ML inference is handled by the Kaggle T4 backend.
# Final image size: ~80 MB

FROM python:3.11-slim

WORKDIR /app

# Only copy what the gateway needs
COPY render_app/requirements.txt .

# Install lightweight deps only (no torch, no transformers)
RUN pip install --no-cache-dir -r requirements.txt

# Copy gateway code + frontend assets
COPY render_app/ ./render_app/
COPY templates/  ./templates/
COPY static/     ./static/

EXPOSE 8000

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT:-8000} --timeout 120 render_app.app:app"]
