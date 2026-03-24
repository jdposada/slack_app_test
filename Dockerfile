FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py ingest.py omop_index.py ./
RUN mkdir -p /app/data \
    && python ingest.py --output /app/data/omop54.db

ENV PORT=8080
ENV OMOP_INDEX_PATH=/app/data/omop54.db

EXPOSE 8080

RUN adduser --disabled-password --gecos "" appuser
USER appuser

CMD ["gunicorn", \
     "--workers", "1", \
     "--threads", "8", \
     "--bind", "0.0.0.0:8080", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:flask_app"]
