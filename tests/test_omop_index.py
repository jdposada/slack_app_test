from pathlib import Path

from omop_index import build_database, parse_document, OmopIndex


def load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text()


def build_index(tmp_path: Path) -> OmopIndex:
    chunks = []
    chunks.extend(
        parse_document(
            load_fixture("cdm54_sample.html"),
            "https://ohdsi.github.io/CommonDataModel/cdm54.html",
        )
    )
    chunks.extend(
        parse_document(
            load_fixture("cdm54_changes_sample.html"),
            "https://ohdsi.github.io/CommonDataModel/cdm54Changes.html",
        )
    )
    db_path = tmp_path / "omop54.db"
    build_database(chunks, str(db_path))
    return OmopIndex(str(db_path))


def test_parser_extracts_table_and_field_metadata() -> None:
    chunks = parse_document(
        load_fixture("cdm54_sample.html"),
        "https://ohdsi.github.io/CommonDataModel/cdm54.html",
    )

    person_chunks = [chunk for chunk in chunks if chunk.table_name == "person"]
    assert person_chunks

    field_chunk = next(chunk for chunk in chunks if chunk.field_name == "gender_concept_id")
    assert field_chunk.required == "Yes"
    assert field_chunk.foreign_key == "Yes"
    assert field_chunk.fk_table == "CONCEPT"
    assert field_chunk.source_url.endswith("#person")
    assert field_chunk.heading_path.endswith("person")


def test_retrieval_prefers_exact_field_match(tmp_path: Path) -> None:
    index = build_index(tmp_path)
    results = index.search("Is gender_concept_id required?")

    assert results
    top = results[0]
    assert top.chunk.field_name == "gender_concept_id"
    assert top.is_confident is True


def test_retrieval_finds_changes_page(tmp_path: Path) -> None:
    index = build_index(tmp_path)
    results = index.search("What changed in visit_occurrence in 5.4?")

    assert results
    assert any("change" in " ".join(result.reasons) or "Changes" in (result.chunk.section_name or "") for result in results)
    assert results[0].chunk.source_url.endswith("#visit_occurrence")


def test_off_topic_query_returns_no_confident_hit(tmp_path: Path) -> None:
    index = build_index(tmp_path)
    results = index.search("What is the capital of France?")

    assert results == [] or results[0].is_confident is False
