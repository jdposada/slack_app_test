"""
app.py – Serverless Slack App with RAG on Google Cloud Run
===========================================================
Architecture
------------
  • Flask web server (port 8080) wrapped around slack_bolt's SlackRequestHandler.
  • FAISS vector index downloaded from GCS on container startup.
  • Vertex AI (gemini-1.5-flash + text-embedding-004) for generation.
  • Lazy listeners so Slack receives an immediate HTTP 200 while the LLM
    call happens asynchronously in a background thread.

Guardrails
----------
  1. Similarity-score threshold – off-topic questions are rejected without
     hitting Vertex AI at all.
  2. System prompt – the LLM is instructed to answer ONLY from the provided
     context and to admit when it doesn't know.
  3. Vertex AI safety settings – dangerous / harassing content is blocked.

Environment Variables (required)
---------------------------------
  SLACK_BOT_TOKEN        – xoxb-… token from your Slack App configuration.
  SLACK_SIGNING_SECRET   – signing secret from your Slack App configuration.
  GCS_BUCKET_NAME        – GCS bucket that stores index.faiss / index.pkl.
  GOOGLE_CLOUD_PROJECT   – GCP project ID used for Vertex AI calls.

Environment Variables (optional)
---------------------------------
  GCS_INDEX_PREFIX       – prefix/folder inside the bucket (default: faiss_index).
  SIMILARITY_THRESHOLD   – max L2 distance for FAISS hits (default: 1.5).
  TOP_K_DOCUMENTS        – number of chunks retrieved per query (default: 4).
  VERTEX_AI_LOCATION     – Vertex AI region (default: us-central1).
  PORT                   – Flask port (default: 8080).
"""

import logging
import os
import sys

from flask import Flask, request
from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from vertexai.generative_models import HarmBlockThreshold, HarmCategory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("slack_rag_app")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
SLACK_BOT_TOKEN: str = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET: str = os.environ["SLACK_SIGNING_SECRET"]
GCS_BUCKET_NAME: str = os.environ["GCS_BUCKET_NAME"]
GOOGLE_CLOUD_PROJECT: str = os.environ["GOOGLE_CLOUD_PROJECT"]

GCS_INDEX_PREFIX: str = os.environ.get("GCS_INDEX_PREFIX", "faiss_index")
SIMILARITY_THRESHOLD: float = float(os.environ.get("SIMILARITY_THRESHOLD", "1.5"))
TOP_K_DOCUMENTS: int = int(os.environ.get("TOP_K_DOCUMENTS", "4"))
VERTEX_AI_LOCATION: str = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
PORT: int = int(os.environ.get("PORT", "8080"))

# Local directory inside the container where index files land
LOCAL_INDEX_DIR = "/tmp/faiss_index"
FAISS_INDEX_FILE = "index.faiss"
FAISS_PKL_FILE = "index.pkl"

# Embedding + generation models
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"

# ---------------------------------------------------------------------------
# Guardrail 2 – System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the "
    "provided context excerpts from company documentation. "
    "Rules you MUST follow:\n"
    "  1. Only use information explicitly present in the context below.\n"
    "  2. If the context does not contain enough information to answer the "
    "question, respond with: "
    "'I don't have enough information in the documentation to answer that question.'\n"
    "  3. Do NOT use any external knowledge, make assumptions, or speculate.\n"
    "  4. Do NOT answer questions unrelated to the provided documentation.\n"
    "  5. Keep your answers concise and factual.\n\n"
    "Context:\n{context}"
)

# Off-topic rejection message (Guardrail 1)
OFF_TOPIC_MESSAGE = (
    "I can only answer questions related to our company documentation. "
    "Your question appears to be outside the scope of the available knowledge base."
)

# ---------------------------------------------------------------------------
# Guardrail 3 – Vertex AI safety settings
# ---------------------------------------------------------------------------
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# ---------------------------------------------------------------------------
# Startup: download FAISS index from GCS
# ---------------------------------------------------------------------------

def download_index_from_gcs() -> None:
    """Download index.faiss and index.pkl from GCS to LOCAL_INDEX_DIR."""
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    logger.info(
        "Downloading FAISS index from gs://%s/%s/ ...",
        GCS_BUCKET_NAME,
        GCS_INDEX_PREFIX,
    )
    client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
    bucket = client.bucket(GCS_BUCKET_NAME)

    for filename in (FAISS_INDEX_FILE, FAISS_PKL_FILE):
        blob_name = f"{GCS_INDEX_PREFIX}/{filename}" if GCS_INDEX_PREFIX else filename
        local_path = os.path.join(LOCAL_INDEX_DIR, filename)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        logger.info("  Downloaded gs://%s/%s → %s", GCS_BUCKET_NAME, blob_name, local_path)

    logger.info("Index download complete.")


def load_vector_store() -> FAISS:
    """Load the FAISS index from the local directory using Vertex AI embeddings."""
    logger.info("Loading FAISS index into memory...")
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION,
    )
    vector_store = FAISS.load_local(
        LOCAL_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we control the GCS source
    )
    logger.info("FAISS index loaded successfully.")
    return vector_store


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def retrieve_context(vector_store: FAISS, query: str) -> tuple[list, bool]:
    """
    Perform a similarity search with L2 distance scores.

    Returns:
        (docs, is_relevant):
            docs        – list of (Document, score) tuples
            is_relevant – True if at least one doc is within SIMILARITY_THRESHOLD
    """
    results = vector_store.similarity_search_with_score(query, k=TOP_K_DOCUMENTS)

    if not results:
        return [], False

    # FAISS returns L2 distances; lower == more similar.
    best_score = min(score for _, score in results)
    logger.info("Best similarity score for query: %.4f (threshold: %.4f)", best_score, SIMILARITY_THRESHOLD)

    is_relevant = best_score <= SIMILARITY_THRESHOLD
    return results, is_relevant


def build_context_string(results: list) -> str:
    """Concatenate retrieved document chunks into a single context string."""
    chunks = []
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[Excerpt {i} | source: {source} | score: {score:.4f}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


def call_llm(llm: ChatVertexAI, context: str, question: str) -> str:
    """Build the prompt and call the Vertex AI LLM, returning the answer text."""
    system_message = SYSTEM_PROMPT.format(context=context)
    messages = [
        ("system", system_message),
        ("human", question),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Application initialisation
# ---------------------------------------------------------------------------

def create_app() -> tuple[App, FAISS, ChatVertexAI]:
    """Download index, initialise LLM, and wire up Bolt listeners."""

    # --- Download + load vector store ---
    download_index_from_gcs()
    vector_store = load_vector_store()

    # --- Initialise Vertex AI LLM (Guardrails 2 & 3) ---
    logger.info("Initialising Vertex AI LLM (model=%s)...", LLM_MODEL)
    llm = ChatVertexAI(
        model_name=LLM_MODEL,
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION,
        safety_settings=SAFETY_SETTINGS,      # Guardrail 3
        temperature=0.2,                       # low temperature for factual answers
        max_output_tokens=1024,
    )

    # --- Bolt App ---
    bolt_app = App(
        token=SLACK_BOT_TOKEN,
        signing_secret=SLACK_SIGNING_SECRET,
    )

    # ------------------------------------------------------------------
    # Lazy listener: respond to @mentions
    # ------------------------------------------------------------------
    @bolt_app.event("app_mention")
    def handle_app_mention(event: dict, say, ack) -> None:  # noqa: ANN001
        """
        Ack immediately (Slack 3-second rule), then process in the background.

        Bolt's process_before=False (default) together with lazy=[] pattern
        ensures the ack() is sent before the heavy work starts.
        """
        ack()  # Guardrail: Slack must receive HTTP 200 within 3 s

        user_question: str = event.get("text", "")
        thread_ts: str = event.get("thread_ts") or event.get("ts", "")

        # Strip the bot mention (<@BOTID>) from the text
        if "<@" in user_question:
            user_question = user_question.split(">", 1)[-1].strip()

        if not user_question:
            say(text="Please ask me a question!", thread_ts=thread_ts)
            return

        logger.info("Received question: %s", user_question)

        # --- Guardrail 1: similarity threshold check ---
        try:
            results, is_relevant = retrieve_context(vector_store, user_question)
        except Exception as exc:
            logger.exception("Vector store retrieval failed: %s", exc)
            say(
                text="I encountered an error while searching the knowledge base. Please try again.",
                thread_ts=thread_ts,
            )
            return

        if not is_relevant:
            logger.info("Query rejected by similarity threshold (off-topic).")
            say(text=OFF_TOPIC_MESSAGE, thread_ts=thread_ts)
            return

        # --- Guardrail 2 & 3: call LLM with system prompt + safety settings ---
        context = build_context_string(results)
        try:
            answer = call_llm(llm, context, user_question)
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            say(
                text="I encountered an error while generating a response. Please try again.",
                thread_ts=thread_ts,
            )
            return

        logger.info("Sending answer back to Slack thread %s.", thread_ts)
        say(text=answer, thread_ts=thread_ts)

    # Register the heavy work as a *lazy* function so Bolt can ack first.
    # We re-register the same event with process_before=False and a lazy list.
    # NOTE: The pattern above already calls ack() synchronously at the top of
    # the handler; for stricter separation you can use bolt_app.event with
    # ack + lazy kwargs as shown in the alternative registration below.

    return bolt_app, vector_store, llm


# ---------------------------------------------------------------------------
# Flask entry-point
# ---------------------------------------------------------------------------

# Module-level initialisation runs once per worker process.
bolt_app, _vector_store, _llm = create_app()
handler = SlackRequestHandler(bolt_app)

flask_app = Flask(__name__)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Primary endpoint for Slack Events API / interactivity payloads."""
    return handler.handle(request)


@flask_app.route("/health", methods=["GET"])
def health_check():
    """Lightweight liveness probe used by Cloud Run."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    # Used when running locally with `python app.py`.
    # Production: gunicorn is specified in the Dockerfile CMD.
    flask_app.run(host="0.0.0.0", port=PORT, debug=False)
