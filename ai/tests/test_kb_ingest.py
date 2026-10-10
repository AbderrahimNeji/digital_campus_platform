"""Bundled knowledge-base ingestion must not be able to wipe student data."""

from pathlib import Path

from ai.rag.kb_ingest import (
    STALE_CHUNK_DELETE_SQL,
    UPSERT_CHUNK_SQL,
    UPSERT_DOCUMENT_SQL,
    chunk_id,
    document_id,
    missing_chunk_columns,
)

ROOT = Path(__file__).resolve().parents[2]


def test_ids_stay_stable_across_runs():
    assert document_id("guide.md") == document_id("guide.md")
    assert chunk_id("chunk-1") == chunk_id("chunk-1")
    assert document_id("guide.md") != document_id("other.md")
    assert chunk_id("chunk-1") != chunk_id("chunk-2")


def test_stale_delete_is_limited_to_global_bundled_chunks():
    sql = STALE_CHUNK_DELETE_SQL.lower()
    assert "delete from document_chunks" in sql
    assert "student_id is null" in sql
    assert "material_id is null" in sql
    assert "document_id = cast(:doc_id as uuid)" in sql
    for forbidden in (
        "delete from students",
        "delete from courses",
        "delete from course_materials",
        "delete from chat_sessions",
        "delete from chat_messages",
        "delete from documents",
    ):
        assert forbidden not in sql


def test_chunk_upsert_will_not_replace_a_student_row():
    sql = UPSERT_CHUNK_SQL.lower()
    assert "on conflict (chunk_id) do update" in sql
    assert "student_id is null" in sql
    assert "material_id is null" in sql
    assert "delete from" not in sql
    assert "delete from" not in UPSERT_DOCUMENT_SQL.lower()


def test_missing_schema_is_reported_before_any_write():
    assert missing_chunk_columns({"chunk_id", "text"}) == sorted(
        {"document_id", "title", "source", "page", "category", "embedding", "student_id", "material_id"}
    )
    assert missing_chunk_columns({
        "chunk_id", "document_id", "text", "title", "source", "page", "category",
        "embedding", "student_id", "material_id",
    }) == []


def test_ingest_script_does_not_wipe_tables():
    source = (ROOT / "scripts" / "ingest_kb.py").read_text(encoding="utf-8").lower()
    assert "delete from documents" not in source
    assert "delete from document_chunks" not in source
    assert "drop table " not in source
    assert "drop schema" not in source
    assert "create table " not in source
