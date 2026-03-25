import hashlib
import hmac
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import vertexai
from flask import Flask, Response, jsonify, request
from slack_sdk.webhook import WebhookClient
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from omop_index import OmopIndex, SearchHit, extract_query_pairs, extract_query_terms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("omop54_slack_app")


NO_ANSWER_MESSAGE = (
    "I couldn't find enough support for that in the OMOP 5.4 documentation set."
)
WORKING_MESSAGE = "Working on it..."
_REQUEST_TTL_SECONDS = 300
SYSTEM_PROMPT = (
    "You answer questions about OMOP CDM v5.4 using only the provided excerpts from "
    "the official OHDSI CommonDataModel documentation.\n"
    "Rules:\n"
    "1. Use only the supplied excerpts.\n"
    "2. If the excerpts do not answer the question, reply exactly with: "
    "\"I don't have enough information in the OMOP 5.4 documentation to answer that.\"\n"
    "3. Do not rely on outside biomedical or OHDSI knowledge.\n"
    "4. Keep the answer concise and factual.\n"
    "5. Do not include a Sources section in your answer."
)
SAFETY_SETTINGS = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    slack_signing_secret: str
    google_cloud_project: str
    vertex_ai_location: str
    vertex_ai_model: str
    omop_index_path: str
    enable_debug_command: bool
    top_k_documents: int
    source_count: int
    port: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            slack_signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            google_cloud_project=os.environ["GOOGLE_CLOUD_PROJECT"],
            vertex_ai_location=os.environ.get("VERTEX_AI_LOCATION", "us-central1"),
            vertex_ai_model=os.environ.get("VERTEX_AI_MODEL", "gemini-2.5-flash"),
            omop_index_path=os.environ.get("OMOP_INDEX_PATH", "data/omop54.db"),
            enable_debug_command=parse_bool(os.environ.get("ENABLE_DEBUG_COMMAND")),
            top_k_documents=int(os.environ.get("TOP_K_DOCUMENTS", "4")),
            source_count=int(os.environ.get("SOURCE_COUNT", "3")),
            port=int(os.environ.get("PORT", "8080")),
        )


class AnswerGenerator(Protocol):
    def answer(self, question: str, context: str) -> str:
        ...


class VertexAnswerer:
    def __init__(self, settings: Settings):
        vertexai.init(
            project=settings.google_cloud_project,
            location=settings.vertex_ai_location,
        )
        self._model = GenerativeModel(
            settings.vertex_ai_model,
            system_instruction=[SYSTEM_PROMPT],
        )

    def answer(self, question: str, context: str) -> str:
        prompt = (
            f"Question:\n{question}\n\n"
            f"Documentation excerpts:\n{context}\n\n"
            "Answer from the excerpts only."
        )
        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.1,
                max_output_tokens=700,
            ),
            safety_settings=SAFETY_SETTINGS,
        )
        text = getattr(response, "text", "").strip()
        if not text:
            raise ValueError("Vertex AI returned an empty response.")
        return text


class OmopAssistant:
    def __init__(
        self,
        index: OmopIndex,
        answerer: AnswerGenerator,
        *,
        top_k_documents: int = 4,
        source_count: int = 3,
    ):
        self._index = index
        self._answerer = answerer
        self._top_k_documents = top_k_documents
        self._source_count = source_count

    def answer_question(self, question: str, *, debug: bool = False) -> str:
        cleaned_question = question.strip()
        if not cleaned_question:
            return "Ask a question after the command, for example `/omop54 What is PERSON?`."

        hits = self._index.search(cleaned_question, limit=max(self._top_k_documents, 6))
        if not hits or not hits[0].is_confident:
            return self._format_no_answer(hits if debug else [])

        ambiguous_message = self._build_ambiguity_message(cleaned_question, hits, debug=debug)
        if ambiguous_message:
            return ambiguous_message

        context_hits = hits[: self._top_k_documents]
        answer = self._answerer.answer(cleaned_question, build_context(context_hits))
        return self._format_answer(answer, hits, debug=debug)

    def _format_no_answer(self, hits: list[SearchHit]) -> str:
        if not hits:
            return NO_ANSWER_MESSAGE
        lines = [NO_ANSWER_MESSAGE, "", "Closest matches:"]
        for hit in hits[:3]:
            lines.append(f"- {format_source_line(hit)}")
        return "\n".join(lines)

    def _format_answer(self, answer: str, hits: list[SearchHit], *, debug: bool) -> str:
        lines = [answer.strip(), "", "Sources:"]
        seen_urls: set[str] = set()
        emitted = 0
        for hit in hits:
            url = hit.chunk.source_url
            if url in seen_urls:
                continue
            seen_urls.add(url)
            lines.append(f"- {format_source_line(hit)}")
            emitted += 1
            if emitted >= self._source_count:
                break

        if debug:
            lines.extend(["", "Debug diagnostics:"])
            for hit in hits[: self._top_k_documents]:
                reason_text = ", ".join(hit.reasons)
                if hit.chunk.field_name:
                    target = f"field {hit.chunk.field_name}"
                elif hit.chunk.table_name:
                    target = f"table {hit.chunk.table_name.upper()}"
                else:
                    target = "n/a"
                lines.append(
                    f"- score={hit.score:.1f}; overlap={hit.token_overlap}; "
                    f"reason={reason_text}; target={target}; source={hit.chunk.source_url}"
                )
                lines.append(f"  excerpt={hit.chunk.body[:280].replace(chr(10), ' ')}")

        return "\n".join(lines)

    def _build_ambiguity_message(
        self,
        question: str,
        hits: list[SearchHit],
        *,
        debug: bool,
    ) -> str | None:
        if extract_query_pairs(question):
            return None

        candidate_fields = [term for term in extract_query_terms(question) if "_" in term]
        if not candidate_fields:
            return None

        for field_name in candidate_fields:
            tables = [
                hit.chunk.table_name
                for hit in hits
                if hit.chunk.field_name == field_name
                and "exact field match" in hit.reasons
                and hit.chunk.table_name
            ]
            unique_tables = sorted({table for table in tables if table})
            if len(unique_tables) > 1:
                example_table = unique_tables[0].upper()
                table_list = ", ".join(table.upper() for table in unique_tables)
                message = (
                    f"`{field_name}` appears in multiple OMOP tables: {table_list}. "
                    f"Please ask again with a table name, for example "
                    f"`/omop54 Is {example_table}.{field_name} required?`"
                )
                if not debug:
                    return message

                lines = [message, "", "Debug diagnostics:"]
                for hit in hits[: self._top_k_documents]:
                    reason_text = ", ".join(hit.reasons)
                    if hit.chunk.field_name:
                        target = f"field {hit.chunk.field_name}"
                    elif hit.chunk.table_name:
                        target = f"table {hit.chunk.table_name.upper()}"
                    else:
                        target = "n/a"
                    lines.append(
                        f"- score={hit.score:.1f}; overlap={hit.token_overlap}; "
                        f"reason={reason_text}; target={target}; source={hit.chunk.source_url}"
                    )
                    lines.append(f"  excerpt={hit.chunk.body[:280].replace(chr(10), ' ')}")
                return "\n".join(lines)

        return None


def build_context(hits: list[SearchHit]) -> str:
    excerpts = []
    for index, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        label_parts = [chunk.page_title]
        if chunk.table_name:
            label_parts.append(f"table={chunk.table_name.upper()}")
        if chunk.field_name:
            label_parts.append(f"field={chunk.field_name}")
        if chunk.section_name:
            label_parts.append(f"section={chunk.section_name}")
        label_parts.append(f"url={chunk.source_url}")
        excerpts.append(
            f"[Excerpt {index} | {' | '.join(label_parts)}]\n{chunk.body[:1800]}"
        )
    return "\n\n---\n\n".join(excerpts)


def format_source_line(hit: SearchHit) -> str:
    chunk = hit.chunk
    label_parts = [chunk.page_title]
    if chunk.table_name:
        label_parts.append(f"table {chunk.table_name.upper()}")
    if chunk.field_name:
        label_parts.append(f"field {chunk.field_name}")
    if chunk.section_name:
        label_parts.append(chunk.section_name)
    return f"{' | '.join(label_parts)} | {chunk.source_url}"


def send_ephemeral_response(
    response_url: str,
    text: str,
    *,
    replace_original: bool,
) -> None:
    webhook = WebhookClient(response_url)
    webhook.send(
        text=text,
        response_type="ephemeral",
        replace_original=replace_original,
    )


class RequestTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, float] = {}

    def mark_if_new(self, request_id: str) -> bool:
        now = time.time()
        with self._lock:
            expired = [
                seen_id
                for seen_id, seen_at in self._seen.items()
                if now - seen_at > _REQUEST_TTL_SECONDS
            ]
            for seen_id in expired:
                self._seen.pop(seen_id, None)

            if request_id in self._seen:
                return False

            self._seen[request_id] = now
            return True


def build_working_response() -> Response:
    return jsonify(
        {
            "response_type": "ephemeral",
            "text": WORKING_MESSAGE,
        }
    )


def verify_slack_request(signing_secret: str, *, timestamp: str, signature: str, body: bytes) -> bool:
    try:
        request_time = int(timestamp)
    except (TypeError, ValueError):
        return False

    if abs(time.time() - request_time) > _REQUEST_TTL_SECONDS:
        return False

    payload = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def build_request_id(command_name: str, trigger_id: str, body: bytes) -> str:
    if trigger_id:
        return trigger_id
    return f"{command_name}:{hashlib.sha256(body).hexdigest()}"


def launch_worker(question: str, response_url: str, debug: bool) -> None:
    def run() -> None:
        try:
            assistant = initialize_runtime()
            text = assistant.answer_question(question, debug=debug)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Failed to process Slack command: %s", exc)
            text = "I hit an internal error while answering that OMOP 5.4 question."
        send_ephemeral_response(
            response_url,
            text,
            replace_original=True,
        )

    threading.Thread(target=run, daemon=True).start()


def create_runtime(settings: Settings) -> OmopAssistant:
    if not os.path.exists(settings.omop_index_path):
        raise FileNotFoundError(
            f"OMOP index not found at {settings.omop_index_path}. Run ingest.py first."
        )

    index = OmopIndex(settings.omop_index_path)
    return OmopAssistant(
        index=index,
        answerer=VertexAnswerer(settings),
        top_k_documents=settings.top_k_documents,
        source_count=settings.source_count,
    )


flask_app = Flask(__name__)
_handler_lock = threading.Lock()
SETTINGS: Settings | None = None
ASSISTANT: OmopAssistant | None = None
REQUEST_TRACKER = RequestTracker()

def initialize_runtime() -> OmopAssistant:
    global SETTINGS, ASSISTANT
    if ASSISTANT is not None:
        return ASSISTANT

    with _handler_lock:
        if ASSISTANT is None:
            logger.info("Initialising OMOP 5.4 Slack runtime.")
            SETTINGS = Settings.from_env()
            ASSISTANT = create_runtime(SETTINGS)
    return ASSISTANT


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    settings = SETTINGS or Settings.from_env()
    raw_body = request.get_data(cache=True)
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    if not verify_slack_request(
        settings.slack_signing_secret,
        timestamp=timestamp,
        signature=signature,
        body=raw_body,
    ):
        return {"error": "invalid request signature"}, 401

    if request.form.get("ssl_check") == "1":
        return {"ok": True}, 200

    command_name = request.form.get("command", "")
    if command_name not in {"/omop54", "/omop54-debug"}:
        return {"error": "unsupported command"}, 404

    if command_name == "/omop54-debug" and not settings.enable_debug_command:
        return jsonify(
            {
                "response_type": "ephemeral",
                "text": "The debug command is disabled for this deployment.",
            }
        )

    request_id = build_request_id(
        command_name,
        request.form.get("trigger_id", ""),
        raw_body,
    )
    if REQUEST_TRACKER.mark_if_new(request_id):
        launch_worker(
            question=request.form.get("text", ""),
            response_url=request.form["response_url"],
            debug=command_name == "/omop54-debug",
        )

    return build_working_response()


@flask_app.route("/health", methods=["GET"])
def health_check():
    if ASSISTANT is None:
        return {"status": "starting"}, 503
    return {"status": "ok"}, 200


if __name__ == "__main__":
    settings = SETTINGS or Settings.from_env()
    initialize_runtime()
    flask_app.run(host="0.0.0.0", port=settings.port, debug=False)
