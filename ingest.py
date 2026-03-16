"""
ingest.py – Data Ingestion Script
===================================
Run this script *locally* (or in a CI job) to:
  1. Load documents from a public URL (WebBaseLoader) and a local PDF (PyPDFLoader).
  2. Split the text into overlapping chunks.
  3. Embed the chunks with Vertex AI text-embedding-004.
  4. Build a FAISS vector store and persist it to disk (index.faiss + index.pkl).
  5. Upload both files to Google Cloud Storage so Cloud Run can download them at startup.

Prerequisites
-------------
* Google Cloud project with Vertex AI API enabled.
* Application Default Credentials configured  (`gcloud auth application-default login`).
* Environment variables (or edit the CONFIG section below):
    GCS_BUCKET_NAME      – target GCS bucket (must already exist)
    GOOGLE_CLOUD_PROJECT – GCP project ID
    GCS_INDEX_PREFIX     – optional path prefix inside the bucket (default: "faiss_index")

Usage
-----
    python ingest.py --pdf /path/to/your.pdf --url https://example.com/docs
"""

import argparse
import logging
import os
import sys
import tempfile

from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# CONFIG  (override via environment variables or CLI flags)
# ---------------------------------------------------------------------------
GCS_BUCKET_NAME: str = os.environ.get("GCS_BUCKET_NAME", "")
GOOGLE_CLOUD_PROJECT: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
GCS_INDEX_PREFIX: str = os.environ.get("GCS_INDEX_PREFIX", "faiss_index")

# FAISS file names as produced by LangChain's FAISS.save_local()
FAISS_INDEX_FILE = "index.faiss"
FAISS_PKL_FILE = "index.pkl"

# Embedding model hosted on Vertex AI
EMBEDDING_MODEL = "text-embedding-004"

# Chunking parameters – tune these for your document corpus
CHUNK_SIZE = 1_000       # characters per chunk
CHUNK_OVERLAP = 200      # overlap to preserve sentence continuity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_documents(pdf_path: str, web_url: str) -> list:
    """Load documents from a local PDF and a public URL."""
    documents = []

    # --- Web page ---
    if web_url:
        logger.info("Loading web page: %s", web_url)
        try:
            web_loader = WebBaseLoader(web_url)
            web_docs = web_loader.load()
            logger.info("  Loaded %d document(s) from URL.", len(web_docs))
            documents.extend(web_docs)
        except Exception as exc:
            logger.error("Failed to load URL '%s': %s", web_url, exc)
            raise

    # --- PDF ---
    if pdf_path:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        logger.info("Loading PDF: %s", pdf_path)
        try:
            pdf_loader = PyPDFLoader(pdf_path)
            pdf_docs = pdf_loader.load()
            logger.info("  Loaded %d page(s) from PDF.", len(pdf_docs))
            documents.extend(pdf_docs)
        except Exception as exc:
            logger.error("Failed to load PDF '%s': %s", pdf_path, exc)
            raise

    if not documents:
        raise ValueError("No documents were loaded. Provide at least one --pdf or --url.")

    return documents


def split_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks (size=%d, overlap=%d).", len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)
    return chunks


def build_faiss_index(chunks: list, project: str, output_dir: str) -> None:
    """Embed chunks and persist a FAISS index to *output_dir*."""
    logger.info("Initialising Vertex AI embeddings (model=%s, project=%s)...", EMBEDDING_MODEL, project)
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=project,
    )

    logger.info("Building FAISS index from %d chunks – this may take a few minutes...", len(chunks))
    vector_store = FAISS.from_documents(chunks, embeddings)

    logger.info("Saving FAISS index to '%s'...", output_dir)
    vector_store.save_local(output_dir)
    logger.info("Index saved: %s, %s", FAISS_INDEX_FILE, FAISS_PKL_FILE)


def upload_to_gcs(local_dir: str, bucket_name: str, prefix: str) -> None:
    """Upload FAISS index files from *local_dir* to GCS under *prefix*."""
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME is required for upload. Set the env var or use --bucket.")

    logger.info("Uploading index files to gs://%s/%s/ ...", bucket_name, prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for filename in (FAISS_INDEX_FILE, FAISS_PKL_FILE):
        local_path = os.path.join(local_dir, filename)
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Expected FAISS file not found: {local_path}")
        blob_name = f"{prefix}/{filename}" if prefix else filename
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        logger.info("  Uploaded %s → gs://%s/%s", filename, bucket_name, blob_name)

    logger.info("Upload complete.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest documents into a FAISS index and upload to GCS.",
    )
    parser.add_argument(
        "--pdf",
        default="",
        metavar="PATH",
        help="Path to a local PDF file to ingest.",
    )
    parser.add_argument(
        "--url",
        default="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        metavar="URL",
        help="Public URL to load with WebBaseLoader.",
    )
    parser.add_argument(
        "--bucket",
        default=GCS_BUCKET_NAME,
        metavar="BUCKET",
        help="GCS bucket name (overrides GCS_BUCKET_NAME env var).",
    )
    parser.add_argument(
        "--project",
        default=GOOGLE_CLOUD_PROJECT,
        metavar="PROJECT",
        help="GCP project ID (overrides GOOGLE_CLOUD_PROJECT env var).",
    )
    parser.add_argument(
        "--prefix",
        default=GCS_INDEX_PREFIX,
        metavar="PREFIX",
        help="GCS path prefix for the index files (default: faiss_index).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        metavar="DIR",
        help=(
            "Local directory to save the index. "
            "Defaults to a temporary directory that is cleaned up after upload."
        ),
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build the index locally but do not upload to GCS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.project:
        logger.error(
            "GCP project ID is required. Set GOOGLE_CLOUD_PROJECT or use --project."
        )
        sys.exit(1)

    if not args.pdf and not args.url:
        logger.error("At least one of --pdf or --url must be provided.")
        sys.exit(1)

    # Determine output directory
    tmp_dir_handle = None
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        tmp_dir_handle = tempfile.TemporaryDirectory()
        output_dir = tmp_dir_handle.name

    try:
        # Step 1 – Load
        documents = load_documents(pdf_path=args.pdf, web_url=args.url)

        # Step 2 – Chunk
        chunks = split_documents(documents)

        # Step 3 & 4 – Embed + save
        build_faiss_index(chunks, project=args.project, output_dir=output_dir)

        # Step 5 – Upload
        if not args.skip_upload:
            upload_to_gcs(
                local_dir=output_dir,
                bucket_name=args.bucket,
                prefix=args.prefix,
            )
        else:
            logger.info("Skipping GCS upload (--skip-upload flag set). Index is at: %s", output_dir)

    finally:
        if tmp_dir_handle is not None:
            tmp_dir_handle.cleanup()

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
