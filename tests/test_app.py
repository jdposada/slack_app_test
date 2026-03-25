import hashlib
import hmac
import time
from urllib.parse import urlencode
from pathlib import Path

import app as app_module
from app import OmopAssistant, RequestTracker, Settings, WORKING_MESSAGE
from omop_index import OmopIndex, build_database, parse_document


class FakeAnswerer:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, context: str) -> str:
        self.calls += 1
        assert question
        assert context
        return "PERSON stores unique people records."


def build_test_index(tmp_path: Path) -> OmopIndex:
    fixture_dir = Path(__file__).parent / "fixtures"
    chunks = []
    chunks.extend(
        parse_document(
            (fixture_dir / "cdm54_sample.html").read_text(),
            "https://ohdsi.github.io/CommonDataModel/cdm54.html",
        )
    )
    chunks.extend(
        parse_document(
            (fixture_dir / "cdm54_changes_sample.html").read_text(),
            "https://ohdsi.github.io/CommonDataModel/cdm54Changes.html",
        )
    )
    db_path = tmp_path / "omop54.db"
    build_database(chunks, str(db_path))
    return OmopIndex(str(db_path))


def test_answer_question_formats_sources(tmp_path: Path) -> None:
    index = build_test_index(tmp_path)
    answerer = FakeAnswerer()
    assistant = OmopAssistant(index, answerer, top_k_documents=4, source_count=2)

    response = assistant.answer_question("What is PERSON?")

    assert "PERSON stores unique people records." in response
    assert "Sources:" in response
    assert "table PERSON" in response
    assert "https://ohdsi.github.io/CommonDataModel/cdm54.html#person" in response
    assert answerer.calls == 1


def test_debug_response_includes_diagnostics(tmp_path: Path) -> None:
    index = build_test_index(tmp_path)
    assistant = OmopAssistant(index, FakeAnswerer(), top_k_documents=4, source_count=2)

    response = assistant.answer_question("Is gender_concept_id required?", debug=True)

    assert "Debug diagnostics:" in response
    assert "exact field match" in response
    assert "field gender_concept_id" in response


def test_no_answer_skips_model(tmp_path: Path) -> None:
    index = build_test_index(tmp_path)
    answerer = FakeAnswerer()
    assistant = OmopAssistant(index, answerer, top_k_documents=4, source_count=2)

    response = assistant.answer_question("What is the weather today?")

    assert "OMOP 5.4 documentation set" in response
    assert answerer.calls == 0


def test_ambiguous_field_query_requests_table_name(tmp_path: Path) -> None:
    index = build_test_index(tmp_path)
    answerer = FakeAnswerer()
    assistant = OmopAssistant(index, answerer, top_k_documents=4, source_count=2)

    response = assistant.answer_question("Is gender_concept_id required?")

    assert "appears in multiple OMOP tables" in response
    assert "PERSON.gender_concept_id" in response
    assert answerer.calls == 0


def sign_slack_request(secret: str, timestamp: str, body: str) -> str:
    payload = f"v0:{timestamp}:{body}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"v0={digest}"


def test_slack_command_returns_working_message_and_launches_worker(monkeypatch) -> None:
    settings = Settings(
        slack_signing_secret="test-secret",
        google_cloud_project="test-project",
        vertex_ai_location="us-central1",
        vertex_ai_model="gemini-2.5-flash",
        omop_index_path="unused.db",
        enable_debug_command=True,
        top_k_documents=4,
        source_count=3,
        port=8080,
    )
    launched: list[tuple[str, str, str, bool]] = []

    monkeypatch.setattr(app_module, "SETTINGS", settings)
    monkeypatch.setattr(app_module, "REQUEST_TRACKER", RequestTracker())
    monkeypatch.setattr(
        app_module,
        "launch_worker",
        lambda question, response_url, debug: launched.append((question, response_url, debug)),
    )

    body = urlencode(
        {
            "command": "/omop54",
            "text": "What is PERSON?",
            "response_url": "https://example.com/respond",
            "trigger_id": "trigger-1",
        }
    )
    timestamp = str(int(time.time()))
    response = app_module.flask_app.test_client().post(
        "/slack/events",
        data=body,
        content_type="application/x-www-form-urlencoded",
        headers={
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": sign_slack_request(settings.slack_signing_secret, timestamp, body),
        },
    )

    assert response.status_code == 200
    assert response.get_json() == {"response_type": "ephemeral", "text": WORKING_MESSAGE}
    assert launched == [("What is PERSON?", "https://example.com/respond", False)]


def test_slack_retry_does_not_launch_worker_twice(monkeypatch) -> None:
    settings = Settings(
        slack_signing_secret="test-secret",
        google_cloud_project="test-project",
        vertex_ai_location="us-central1",
        vertex_ai_model="gemini-2.5-flash",
        omop_index_path="unused.db",
        enable_debug_command=True,
        top_k_documents=4,
        source_count=3,
        port=8080,
    )
    launched: list[str] = []

    monkeypatch.setattr(app_module, "SETTINGS", settings)
    monkeypatch.setattr(app_module, "REQUEST_TRACKER", RequestTracker())
    monkeypatch.setattr(
        app_module,
        "launch_worker",
        lambda question, response_url, debug: launched.append(question),
    )

    body = urlencode(
        {
            "command": "/omop54",
            "text": "What is PERSON?",
            "response_url": "https://example.com/respond",
            "trigger_id": "trigger-duplicate",
        }
    )
    timestamp = str(int(time.time()))
    headers = {
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": sign_slack_request(settings.slack_signing_secret, timestamp, body),
    }
    client = app_module.flask_app.test_client()

    first = client.post(
        "/slack/events",
        data=body,
        content_type="application/x-www-form-urlencoded",
        headers=headers,
    )
    second = client.post(
        "/slack/events",
        data=body,
        content_type="application/x-www-form-urlencoded",
        headers=headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert launched == ["What is PERSON?"]
