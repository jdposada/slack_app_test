import logging
import os
import sys
import threading
from dataclasses import dataclass
from typing import Protocol

import vertexai
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_sdk.webhook import WebhookClient
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting,
)

from omop_index import OmopIndex, SearchHit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("omop54_slack_app")


NO_ANSWER_MESSAGE = (
    "I couldn't find enough support for that in the OMOP 5.4 documentation set."
)
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
    slack_bot_token: str
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
            slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
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
                lines.append(
                    f"- score={hit.score:.1f}; overlap={hit.token_overlap}; "
                    f"reason={reason_text}; source={hit.chunk.source_url}"
                )
                lines.append(f"  excerpt={hit.chunk.body[:280].replace(chr(10), ' ')}")

        return "\n".join(lines)


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


def send_ephemeral_response(response_url: str, text: str) -> None:
    webhook = WebhookClient(response_url)
    webhook.send(
        text=text,
        response_type="ephemeral",
        replace_original=False,
    )


def register_slack_commands(
    bolt_app: App,
    assistant: OmopAssistant,
    *,
    enable_debug_command: bool,
) -> None:
    def launch_worker(question: str, response_url: str, debug: bool) -> None:
        def run() -> None:
            try:
                text = assistant.answer_question(question, debug=debug)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                logger.exception("Failed to process Slack command: %s", exc)
                text = "I hit an internal error while answering that OMOP 5.4 question."
            send_ephemeral_response(response_url, text)

        threading.Thread(target=run, daemon=True).start()

    @bolt_app.command("/omop54")
    def omop54_command(ack, command) -> None:  # noqa: ANN001
        ack()
        launch_worker(
            question=command.get("text", ""),
            response_url=command["response_url"],
            debug=False,
        )

    if enable_debug_command:

        @bolt_app.command("/omop54-debug")
        def omop54_debug_command(ack, command) -> None:  # noqa: ANN001
            ack()
            launch_worker(
                question=command.get("text", ""),
                response_url=command["response_url"],
                debug=True,
            )


def create_runtime(settings: Settings) -> SlackRequestHandler:
    if not os.path.exists(settings.omop_index_path):
        raise FileNotFoundError(
            f"OMOP index not found at {settings.omop_index_path}. Run ingest.py first."
        )

    index = OmopIndex(settings.omop_index_path)
    assistant = OmopAssistant(
        index=index,
        answerer=VertexAnswerer(settings),
        top_k_documents=settings.top_k_documents,
        source_count=settings.source_count,
    )
    bolt_app = App(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret,
    )
    register_slack_commands(
        bolt_app,
        assistant,
        enable_debug_command=settings.enable_debug_command,
    )
    return SlackRequestHandler(bolt_app)


flask_app = Flask(__name__)
_handler_lock = threading.Lock()
_handler: SlackRequestHandler | None = None


def get_handler() -> SlackRequestHandler:
    global _handler
    if _handler is not None:
        return _handler

    with _handler_lock:
        if _handler is None:
            logger.info("Initialising OMOP 5.4 Slack runtime.")
            _handler = create_runtime(Settings.from_env())
    return _handler


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return get_handler().handle(request)


@flask_app.route("/health", methods=["GET"])
def health_check():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    settings = Settings.from_env()
    flask_app.run(host="0.0.0.0", port=settings.port, debug=False)
