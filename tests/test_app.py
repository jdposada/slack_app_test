from pathlib import Path

from app import OmopAssistant
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
