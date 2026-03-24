import hashlib
import logging
import os
import re
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


logger = logging.getLogger("omop54_index")

BASE_URL = "https://ohdsi.github.io/CommonDataModel/"
PRIMARY_SPEC_URL = urljoin(BASE_URL, "cdm54.html")
ALLOWED_PAGE_NAMES = {
    "cdm54.html",
    "cdm54Changes.html",
    "cdm54erd.html",
    "cdm54ToolingSupport.html",
    "dataModelConventions.html",
    "cdmPrivacy.html",
    "customConcepts.html",
    "download.html",
    "cdmRPackage.html",
    "drug_dose.html",
    "cdmDecisionTree.html",
    "faq.html",
    "sqlScripts.html",
    "contribute.html",
}
EXCLUDED_PAGE_NAMES = {
    "cdm30.html",
    "cdm53.html",
    "index.html",
}
SECTION_HEADINGS = ("h1", "h2", "h3", "h4")
STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "about",
    "are",
    "can",
    "cdm",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "omop",
    "question",
    "table",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "v5",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}
INTENT_BOOSTS = {
    "change": "change",
    "changed": "change",
    "difference": "change",
    "relationship": "relationship",
    "relationships": "relationship",
    "support": "support",
    "tooling": "support",
    "faq": "faq",
    "privacy": "privacy",
    "convention": "convention",
    "conventions": "convention",
    "sql": "sql",
    "dose": "dose",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    page_title: str
    source_url: str
    anchor: str
    heading_path: str
    content_type: str
    table_name: str | None
    field_name: str | None
    section_name: str | None
    required: str | None
    primary_key: str | None
    foreign_key: str | None
    fk_table: str | None
    body: str
    searchable_text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    reasons: tuple[str, ...]
    token_overlap: int
    fts_rank: float | None
    is_confident: bool


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_symbol(text: str | None) -> str:
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return normalized.strip("_")


def looks_like_table_name(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or " " in cleaned:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", cleaned))


def is_allowed_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc.endswith("ohdsi.github.io"):
        return False
    if not parsed.path.startswith("/CommonDataModel/"):
        return False
    page_name = os.path.basename(parsed.path)
    if page_name in EXCLUDED_PAGE_NAMES:
        return False
    return page_name in ALLOWED_PAGE_NAMES


def fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def discover_urls(session: requests.Session) -> list[str]:
    html = fetch_html(session, PRIMARY_SPEC_URL)
    soup = BeautifulSoup(html, "html.parser")
    urls = {PRIMARY_SPEC_URL}

    for anchor in soup.select("a[href]"):
        candidate = urljoin(PRIMARY_SPEC_URL, anchor["href"])
        candidate = candidate.split("#", 1)[0]
        if is_allowed_url(candidate):
            urls.add(candidate)

    return sorted(urls)


def build_default_index(output_path: str) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "omop54-slack-indexer/1.0"})
    chunks = crawl_and_parse(session)
    build_database(chunks, output_path)


def crawl_and_parse(session: requests.Session) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    queue: deque[str] = deque(discover_urls(session))

    while queue:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        logger.info("Fetching %s", url)
        try:
            html = fetch_html(session, url)
        except requests.RequestException as exc:
            logger.warning("Skipping %s: %s", url, exc)
            continue
        chunks.extend(parse_document(html, url))

    logger.info("Parsed %d OMOP chunks from %d pages.", len(chunks), len(seen))
    return chunks


def parse_document(html: str, source_url: str) -> list[Chunk]:
    soup = BeautifulSoup(html, "html.parser")
    page_title = extract_page_title(soup)
    sections = [
        section
        for section in soup.select("div.section")
        if section.find(SECTION_HEADINGS, recursive=False)
    ]
    if not sections:
        return parse_fallback_document(soup, source_url, page_title)

    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(parse_section(section, source_url, page_title))
    return chunks


def parse_fallback_document(
    soup: BeautifulSoup,
    source_url: str,
    page_title: str,
) -> list[Chunk]:
    body = normalize_whitespace(soup.get_text(" ", strip=True))
    if not body:
        return []
    return [
        make_chunk(
            page_title=page_title,
            source_url=source_url,
            anchor="",
            heading_path=page_title,
            content_type="page_section",
            table_name=None,
            field_name=None,
            section_name="Summary",
            required=None,
            primary_key=None,
            foreign_key=None,
            fk_table=None,
            body=body,
        )
    ]


def extract_page_title(soup: BeautifulSoup) -> str:
    heading = soup.find("h1")
    if heading:
        return normalize_whitespace(heading.get_text(" ", strip=True))
    if soup.title:
        return normalize_whitespace(soup.title.get_text(" ", strip=True))
    return "OMOP CommonDataModel"


def parse_section(section: Tag, source_url: str, page_title: str) -> list[Chunk]:
    heading = section.find(SECTION_HEADINGS, recursive=False)
    if heading is None:
        return []

    heading_text = normalize_whitespace(heading.get_text(" ", strip=True))
    heading_path = " > ".join(get_heading_path(section))
    anchor = section.get("id") or heading.get("id") or ""
    page_name = os.path.basename(urlparse(source_url).path)
    inferred_table = infer_table_name(heading_text)
    content_blocks: list[tuple[str, str]] = []
    table_tags: list[Tag] = []

    for child in section.children:
        if not isinstance(child, Tag):
            continue
        if child.name in SECTION_HEADINGS:
            continue
        if child.name == "div" and "section" in child.get("class", []):
            continue
        if child.name == "table":
            table_tags.append(child)
            continue
        if child.name == "p":
            text = normalize_whitespace(child.get_text(" ", strip=True))
            if text:
                content_blocks.append(("paragraph", text))
        elif child.name in {"ul", "ol"}:
            items = [
                normalize_whitespace(item.get_text(" ", strip=True))
                for item in child.find_all("li", recursive=False)
            ]
            items = [item for item in items if item]
            if items:
                content_blocks.append(("list", "\n".join(f"- {item}" for item in items)))

    chunks: list[Chunk] = []
    for section_name, body in split_labeled_blocks(content_blocks):
        content_type = "table_section" if inferred_table else "page_section"
        display_section_name = section_name
        lowered_page_name = page_name.lower()
        if "change" in lowered_page_name:
            content_type = "change_entry"
            if display_section_name == "Summary":
                display_section_name = "Changes"
        elif "support" in lowered_page_name:
            content_type = "support_entry"
            if display_section_name == "Summary":
                display_section_name = "Tooling Support"

        chunks.append(
            make_chunk(
                page_title=page_title,
                source_url=with_anchor(source_url, anchor),
                anchor=anchor,
                heading_path=heading_path,
                content_type=content_type,
                table_name=inferred_table,
                field_name=None,
                section_name=display_section_name,
                required=None,
                primary_key=None,
                foreign_key=None,
                fk_table=None,
                body=body,
            )
        )

    for table in table_tags:
        chunks.extend(
            parse_html_table(
                table,
                source_url=with_anchor(source_url, anchor),
                anchor=anchor,
                page_title=page_title,
                heading_path=heading_path,
                section_heading=heading_text,
                inferred_table=inferred_table,
                page_name=page_name,
            )
        )

    return chunks


def infer_table_name(heading_text: str) -> str | None:
    if not looks_like_table_name(heading_text):
        return None
    return normalize_symbol(heading_text)


def get_heading_path(section: Tag) -> list[str]:
    path: list[str] = []
    parent_sections = [
        parent
        for parent in section.parents
        if isinstance(parent, Tag) and parent.name == "div" and "section" in parent.get("class", [])
    ]
    for parent in reversed(parent_sections):
        heading = parent.find(SECTION_HEADINGS, recursive=False)
        if heading:
            text = normalize_whitespace(heading.get_text(" ", strip=True))
            if text:
                path.append(text)
    heading = section.find(SECTION_HEADINGS, recursive=False)
    if heading:
        path.append(normalize_whitespace(heading.get_text(" ", strip=True)))
    return path


def split_labeled_blocks(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not blocks:
        return []

    groups: list[tuple[str, str]] = []
    current_label = "Summary"
    current_parts: list[str] = []

    for block_type, text in blocks:
        if block_type == "paragraph" and is_label_paragraph(text):
            if current_parts:
                groups.append((current_label, "\n".join(current_parts)))
            current_label = text
            current_parts = []
            continue
        current_parts.append(text)

    if current_parts:
        groups.append((current_label, "\n".join(current_parts)))

    return [(label, body) for label, body in groups if body.strip()]


def is_label_paragraph(text: str) -> bool:
    if len(text) > 40:
        return False
    if any(character in text for character in ".:;"):
        return False
    words = text.split()
    if not words:
        return False
    return all(word[:1].isupper() for word in words if word[0].isalpha())


def parse_html_table(
    table: Tag,
    *,
    source_url: str,
    anchor: str,
    page_title: str,
    heading_path: str,
    section_heading: str,
    inferred_table: str | None,
    page_name: str,
) -> list[Chunk]:
    headers = extract_headers(table)
    if not headers:
        return []

    rows = extract_rows(table, len(headers))
    if not rows:
        return []

    if headers[:3] == ["CDM Field", "User Guide", "ETL Conventions"]:
        return [
            make_chunk(
                page_title=page_title,
                source_url=source_url,
                anchor=anchor,
                heading_path=heading_path,
                content_type="field_definition",
                table_name=inferred_table,
                field_name=normalize_symbol(row.get("CDM Field")),
                section_name="Field Specification",
                required=row.get("Required"),
                primary_key=row.get("Primary Key"),
                foreign_key=row.get("Foreign Key"),
                fk_table=row.get("FK Table"),
                body=render_row(headers, row),
            )
            for row in rows
            if row.get("CDM Field")
        ]

    content_type = "table_row"
    section_name = section_heading
    if "change" in page_name.lower():
        content_type = "change_entry"
        section_name = "Changes"
    elif "support" in page_name.lower():
        content_type = "support_entry"
        section_name = "Tooling Support"

    chunks: list[Chunk] = []
    for row in rows:
        body = render_row(headers, row)
        if not body:
            continue
        chunks.append(
            make_chunk(
                page_title=page_title,
                source_url=source_url,
                anchor=anchor,
                heading_path=heading_path,
                content_type=content_type,
                table_name=inferred_table,
                field_name=None,
                section_name=section_name,
                required=None,
                primary_key=None,
                foreign_key=None,
                fk_table=None,
                body=body,
            )
        )
    return chunks


def extract_headers(table: Tag) -> list[str]:
    header_row = table.find("thead")
    cells: list[Tag]
    if header_row:
        first_row = header_row.find("tr")
        cells = first_row.find_all(["th", "td"]) if first_row else []
    else:
        first_row = table.find("tr")
        cells = first_row.find_all(["th", "td"]) if first_row else []
    return [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in cells]


def extract_rows(table: Tag, header_count: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    candidates = table.find_all("tr")
    for row in candidates[1:]:
        cells = [normalize_whitespace(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if not cells:
            continue
        if len(cells) < header_count:
            cells.extend([""] * (header_count - len(cells)))
        elif len(cells) > header_count:
            cells = cells[:header_count]
        header_row = extract_headers(table)
        rows.append(dict(zip(header_row, cells, strict=False)))
    return rows


def render_row(headers: list[str], row: dict[str, str]) -> str:
    parts = []
    for header in headers:
        value = row.get(header, "")
        if value:
            parts.append(f"{header}: {value}")
    return "\n".join(parts)


def with_anchor(source_url: str, anchor: str) -> str:
    if not anchor:
        return source_url
    return f"{source_url.split('#', 1)[0]}#{anchor}"


def make_chunk(
    *,
    page_title: str,
    source_url: str,
    anchor: str,
    heading_path: str,
    content_type: str,
    table_name: str | None,
    field_name: str | None,
    section_name: str | None,
    required: str | None,
    primary_key: str | None,
    foreign_key: str | None,
    fk_table: str | None,
    body: str,
) -> Chunk:
    searchable_text = "\n".join(
        value
        for value in [
            page_title,
            heading_path,
            table_name or "",
            field_name or "",
            section_name or "",
            body,
        ]
        if value
    )
    identity = "|".join(
        [
            source_url,
            content_type,
            table_name or "",
            field_name or "",
            section_name or "",
            body,
        ]
    )
    chunk_id = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        page_title=page_title,
        source_url=source_url,
        anchor=anchor,
        heading_path=heading_path,
        content_type=content_type,
        table_name=table_name,
        field_name=field_name,
        section_name=section_name,
        required=required,
        primary_key=primary_key,
        foreign_key=foreign_key,
        fk_table=fk_table,
        body=body,
        searchable_text=searchable_text,
    )


def build_database(chunks: Iterable[Chunk], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    connection = sqlite3.connect(output)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                page_title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                anchor TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                content_type TEXT NOT NULL,
                table_name TEXT,
                field_name TEXT,
                section_name TEXT,
                required TEXT,
                primary_key_value TEXT,
                foreign_key_value TEXT,
                fk_table TEXT,
                body TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                normalized_table_name TEXT,
                normalized_field_name TEXT,
                searchable_text_norm TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                searchable_text,
                tokenize = 'porter unicode61'
            );

            CREATE INDEX idx_chunks_table_name ON chunks(normalized_table_name);
            CREATE INDEX idx_chunks_field_name ON chunks(normalized_field_name);
            """
        )

        for chunk in chunks:
            normalized_table = normalize_symbol(chunk.table_name)
            normalized_field = normalize_symbol(chunk.field_name)
            searchable_text_norm = normalize_symbol(chunk.searchable_text)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO chunks (
                    chunk_id,
                    page_title,
                    source_url,
                    anchor,
                    heading_path,
                    content_type,
                    table_name,
                    field_name,
                    section_name,
                    required,
                    primary_key_value,
                    foreign_key_value,
                    fk_table,
                    body,
                    searchable_text,
                    normalized_table_name,
                    normalized_field_name,
                    searchable_text_norm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.page_title,
                    chunk.source_url,
                    chunk.anchor,
                    chunk.heading_path,
                    chunk.content_type,
                    chunk.table_name,
                    chunk.field_name,
                    chunk.section_name,
                    chunk.required,
                    chunk.primary_key,
                    chunk.foreign_key,
                    chunk.fk_table,
                    chunk.body,
                    chunk.searchable_text,
                    normalized_table,
                    normalized_field,
                    searchable_text_norm,
                ),
            )
            if cursor.rowcount:
                connection.execute(
                    "INSERT INTO chunks_fts (chunk_id, searchable_text) VALUES (?, ?)",
                    (chunk.chunk_id, chunk.searchable_text),
                )

        connection.commit()
    finally:
        connection.close()


class OmopIndex:
    def __init__(self, db_path: str):
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self._connection.close()

    def search(self, query: str, *, limit: int = 6) -> list[SearchHit]:
        normalized_terms = extract_query_terms(query)
        term_pairs = extract_query_pairs(query)
        candidates: dict[str, dict[str, object]] = {}

        for table_name, field_name in term_pairs:
            for row in self._connection.execute(
                """
                SELECT * FROM chunks
                WHERE normalized_table_name = ? AND normalized_field_name = ?
                """,
                (table_name, field_name),
            ):
                add_candidate(candidates, row, 120.0, "exact table.field match")

        field_terms = [term for term in normalized_terms if "_" in term]
        for field_name in field_terms:
            for row in self._connection.execute(
                "SELECT * FROM chunks WHERE normalized_field_name = ?",
                (field_name,),
            ):
                add_candidate(candidates, row, 90.0, "exact field match")

        for table_name in normalized_terms:
            for row in self._connection.execute(
                """
                SELECT * FROM chunks
                WHERE normalized_table_name = ? AND content_type != 'field_definition'
                """,
                (table_name,),
            ):
                add_candidate(candidates, row, 70.0, "exact table match")

        fts_terms = [term for term in normalized_terms if term not in STOPWORDS]
        if fts_terms:
            fts_query = " OR ".join(f'"{term}"' for term in fts_terms[:8])
            for row in self._connection.execute(
                """
                SELECT c.*, bm25(chunks_fts) AS fts_rank
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT 25
                """,
                (fts_query,),
            ):
                rank = float(row["fts_rank"])
                add_candidate(
                    candidates,
                    row,
                    max(0.0, 35.0 - (rank * 5.0)),
                    "fts lexical match",
                    fts_rank=rank,
                )

        for candidate in candidates.values():
            row = candidate["row"]
            searchable_text_norm = row["searchable_text_norm"]
            overlap = sum(1 for term in normalized_terms if term and term in searchable_text_norm)
            candidate["token_overlap"] = overlap
            candidate["score"] += overlap * 4.0

            intent_bonus = compute_intent_bonus(query, row)
            if intent_bonus:
                candidate["score"] += intent_bonus
                candidate["reasons"].append("intent boost")

            if row["content_type"] == "field_definition":
                candidate["score"] += 8.0
            elif row["content_type"] in {"change_entry", "support_entry"}:
                candidate["score"] += 5.0

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                item["score"],
                item["token_overlap"],
                1 if item["row"]["content_type"] == "field_definition" else 0,
            ),
            reverse=True,
        )

        hits: list[SearchHit] = []
        for candidate in ordered[:limit]:
            row = candidate["row"]
            score = float(candidate["score"])
            overlap = int(candidate.get("token_overlap", 0))
            reasons = tuple(dict.fromkeys(candidate["reasons"]))
            exact_match = any(reason.startswith("exact") for reason in reasons)
            is_confident = exact_match or score >= 45.0 or overlap >= 2
            hits.append(
                SearchHit(
                    chunk=row_to_chunk(row),
                    score=score,
                    reasons=reasons,
                    token_overlap=overlap,
                    fts_rank=candidate.get("fts_rank"),
                    is_confident=is_confident,
                )
            )

        return hits


def extract_query_terms(query: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*", query):
        normalized = normalize_symbol(token)
        if normalized:
            terms.add(normalized)
    return terms


def extract_query_pairs(query: str) -> set[tuple[str, str]]:
    pairs = set()
    for table_name, field_name in re.findall(
        r"([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)",
        query,
    ):
        pairs.add((normalize_symbol(table_name), normalize_symbol(field_name)))
    return pairs


def add_candidate(
    candidates: dict[str, dict[str, object]],
    row: sqlite3.Row,
    score: float,
    reason: str,
    *,
    fts_rank: float | None = None,
) -> None:
    item = candidates.setdefault(
        row["chunk_id"],
        {
            "row": row,
            "score": 0.0,
            "reasons": [],
            "token_overlap": 0,
            "fts_rank": None,
        },
    )
    item["score"] += score
    item["reasons"].append(reason)
    if fts_rank is not None and item["fts_rank"] is None:
        item["fts_rank"] = fts_rank


def compute_intent_bonus(query: str, row: sqlite3.Row) -> float:
    lowered_query = query.lower()
    searchable = f"{row['page_title']} {row['heading_path']} {row['body']}".lower()
    bonus = 0.0
    for token, label in INTENT_BOOSTS.items():
        if token in lowered_query and label in searchable:
            bonus += 12.0
    return bonus


def row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        page_title=row["page_title"],
        source_url=row["source_url"],
        anchor=row["anchor"],
        heading_path=row["heading_path"],
        content_type=row["content_type"],
        table_name=row["table_name"],
        field_name=row["field_name"],
        section_name=row["section_name"],
        required=row["required"],
        primary_key=row["primary_key_value"],
        foreign_key=row["foreign_key_value"],
        fk_table=row["fk_table"],
        body=row["body"],
        searchable_text=row["searchable_text"],
    )
