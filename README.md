# Serverless Slack RAG App

A production-ready Slack App powered by Retrieval-Augmented Generation (RAG), deployed on **Google Cloud Run**.

## Architecture

| Layer | Technology |
|---|---|
| Hosting | Google Cloud Run (serverless) |
| Web framework | Python `flask` + `slack_bolt` (Flask adapter) |
| LLM | Vertex AI `gemini-1.5-flash` |
| Embeddings | Vertex AI `text-embedding-004` |
| Vector store | FAISS (in-memory, index stored in GCS) |
| Object storage | Google Cloud Storage |

---

## Project Structure

```
slack_app_test/
├── app.py            # Cloud Run web server (Slack Bolt + RAG)
├── ingest.py         # Local data ingestion script
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container image definition
└── README.md
```

---

## Component 1 – Data Ingestion (`ingest.py`)

Run this script **locally** (or in a CI job) whenever your documentation changes.

### What it does
1. Loads documents from a public URL (`WebBaseLoader`) and/or a local PDF (`PyPDFLoader`).
2. Splits text into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. Embeds chunks using Vertex AI `text-embedding-004`.
4. Builds a FAISS vector store and saves `index.faiss` + `index.pkl` to disk.
5. Uploads both files to a GCS bucket.

### Prerequisites
* `gcloud auth application-default login` (or a service-account key).
* Vertex AI API enabled in your GCP project.
* A GCS bucket to store the index files.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Minimal – URL only
python ingest.py \
  --url "https://en.wikipedia.org/wiki/Retrieval-augmented_generation" \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET

# URL + local PDF
python ingest.py \
  --url "https://docs.example.com/overview" \
  --pdf  /path/to/internal_handbook.pdf \
  --project YOUR_GCP_PROJECT \
  --bucket YOUR_GCS_BUCKET

# Skip GCS upload (build index locally only)
python ingest.py \
  --url "https://..." \
  --project YOUR_GCP_PROJECT \
  --skip-upload \
  --output-dir ./my_index
```

All flags can also be set via environment variables:

| Flag | Env var | Default |
|---|---|---|
| `--bucket` | `GCS_BUCKET_NAME` | _(required)_ |
| `--project` | `GOOGLE_CLOUD_PROJECT` | _(required)_ |
| `--prefix` | `GCS_INDEX_PREFIX` | `faiss_index` |

---

## Component 2 – Slack App (`app.py`)

### Guardrails

| # | Name | Behaviour |
|---|---|---|
| 1 | Similarity threshold | If the best FAISS L2 distance > `SIMILARITY_THRESHOLD` (default `1.5`), the app replies with a hard-coded "off-topic" message and never calls Vertex AI. |
| 2 | System prompt | The LLM is instructed to answer *only* from the provided context excerpts. |
| 3 | Safety settings | All four Vertex AI `HarmCategory` settings are set to `BLOCK_MEDIUM_AND_ABOVE`. |

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SLACK_BOT_TOKEN` | ✅ | `xoxb-…` bot token |
| `SLACK_SIGNING_SECRET` | ✅ | Signing secret from Slack App settings |
| `GCS_BUCKET_NAME` | ✅ | GCS bucket holding the FAISS index |
| `GOOGLE_CLOUD_PROJECT` | ✅ | GCP project ID |
| `GCS_INDEX_PREFIX` | | Path prefix in the bucket (default: `faiss_index`) |
| `SIMILARITY_THRESHOLD` | | Max L2 distance for relevant hits (default: `1.5`) |
| `TOP_K_DOCUMENTS` | | Chunks retrieved per query (default: `4`) |
| `VERTEX_AI_LOCATION` | | Vertex AI region (default: `us-central1`) |
| `PORT` | | Flask port (default: `8080`) |

### Slack App Setup

1. Go to <https://api.slack.com/apps> and create a new app.
2. **OAuth & Permissions** → add `app_mentions:read`, `chat:write` scopes.
3. **Event Subscriptions** → enable and set the Request URL to `https://<your-cloud-run-url>/slack/events`.
4. Subscribe to the `app_mention` bot event.
5. Install the app to your workspace and copy the **Bot User OAuth Token** and **Signing Secret**.

---

## Component 3 – Deployment

### Build & push the container

```bash
export PROJECT_ID=YOUR_GCP_PROJECT
export REGION=us-central1
export IMAGE="gcr.io/${PROJECT_ID}/slack-rag-app"

# Build
docker build -t "${IMAGE}" .

# Push to Google Container Registry
docker push "${IMAGE}"
```

### Deploy to Cloud Run

```bash
gcloud run deploy slack-rag-app \
  --image        "gcr.io/${PROJECT_ID}/slack-rag-app" \
  --platform     managed \
  --region       "${REGION}" \
  --allow-unauthenticated \
  --port         8080 \
  --cpu          1 \
  --memory       2Gi \
  --timeout      120 \
  --min-instances 1 \
  --set-env-vars "SLACK_BOT_TOKEN=xoxb-YOUR-TOKEN,\
SLACK_SIGNING_SECRET=YOUR-SIGNING-SECRET,\
GCS_BUCKET_NAME=YOUR-GCS-BUCKET,\
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
```

> **Tip:** For production deployments, store secrets in **Google Secret Manager** and reference them with `--set-secrets` instead of plain `--set-env-vars`.

### Grant Cloud Run the necessary IAM permissions

The Cloud Run service identity needs:

```bash
# Allow it to read from the GCS bucket
gcloud storage buckets add-iam-policy-binding "gs://YOUR-GCS-BUCKET" \
  --member "serviceAccount:YOUR-SA@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/storage.objectViewer"

# Allow it to call Vertex AI
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:YOUR-SA@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role "roles/aiplatform.user"
```

### Health Check

Cloud Run automatically probes `GET /health` which returns `{"status": "ok"}`.

---

## Local Development

```bash
# Set required env vars
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_SIGNING_SECRET=...
export GCS_BUCKET_NAME=...
export GOOGLE_CLOUD_PROJECT=...

# Run locally (downloads index from GCS on startup)
python app.py
```

Use [ngrok](https://ngrok.com/) to expose your local server to the Slack Events API during development:

```bash
ngrok http 8080
```
