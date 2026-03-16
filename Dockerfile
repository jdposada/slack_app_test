# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile – Serverless Slack RAG App for Google Cloud Run
# ──────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t slack-rag-app .
# Run:    docker run -p 8080:8080 --env-file .env slack-rag-app
# ──────────────────────────────────────────────────────────────────────────────

# Use the official slim Python 3.11 image as the base.
# Slim variants reduce final image size while keeping pip and standard libs.
FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# build-essential + libgomp1: required to compile / link FAISS native code.
# curl: optional, useful for health-check debugging.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────────────────────
# Copy requirements first to exploit Docker layer caching:
# if requirements.txt is unchanged, pip install is not re-run.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY app.py .

# ── Runtime configuration ────────────────────────────────────────────────────
# Cloud Run injects PORT=8080 automatically; we default to 8080 here.
ENV PORT=8080

# Expose the port for documentation purposes (Cloud Run ignores EXPOSE but
# it is good practice to keep it for local docker run usage).
EXPOSE 8080

# ── Non-root user for security ───────────────────────────────────────────────
# Running as a non-root user follows the principle of least privilege.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# ── Start command ────────────────────────────────────────────────────────────
# gunicorn:
#   --workers 1     – Cloud Run scales via container instances, not workers.
#   --threads 8     – Thread pool handles concurrent requests within one worker.
#   --bind 0.0.0.0  – Listen on all interfaces.
#   --timeout 120   – Allow up to 120 s for LLM + GCS startup calls.
#   --access-logfile – Write access logs to stdout (visible in Cloud Logging).
CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "8", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:flask_app"]
