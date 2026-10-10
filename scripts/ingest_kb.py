"""Upsert the bundled faculty knowledge base into PostgreSQL + pgvector.

Reads ``knowledge_base/build/chunks.jsonl``, embeds each chunk, and stores it
in ``document_chunks``. Safe to run again: document and chunk ids are
deterministic, so existing bundled rows are updated in place.

This script does not create the schema and does not delete student accounts,
courses, course materials, chat history, student chunks, or admin uploads.
Stale rows are removed only when they are global chunks of a bundled document
that is still in this file. Apply Alembic first (``alembic upgrade head``).

Usage, from the repository root, after ``ai/.env`` has ``DATABASE_URL``:

    python scripts/ingest_kb.py

Local development leaves ``EMBEDDING_PROVIDER`` unset (the sentence-transformers
model). To fill a hosted database from this machine without loading PyTorch:

    EMBEDDING_PROVIDER=gemini GEMINI_API_KEY=... DATABASE_URL=... python scripts/ingest_kb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import bindparam, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai.rag.db import get_session  # noqa: E402
from ai.rag.embeddings import EMBEDDING_DIM, embed_texts, embedding_provider  # noqa: E402
from ai.rag.kb_ingest import (  # noqa: E402
    STALE_CHUNK_DELETE_SQL,
    UPSERT_CHUNK_SQL,
    UPSERT_DOCUMENT_SQL,
    chunk_id,
    document_id,
    missing_chunk_columns,
)

CHUNKS_FILE = ROOT / "knowledge_base" / "build" / "chunks.jsonl"
BATCH_SIZE = 64


def _load_rows() -> list[dict]:
    if not CHUNKS_FILE.exists():
        sys.exit(f"Chunks file not found: {CHUNKS_FILE}")
    rows = [
        json.loads(line)
        for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        sys.exit(f"No chunks in {CHUNKS_FILE}")
    return rows


def _require_schema(db) -> None:
    present = set(
        db.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'document_chunks'
        """)).scalars()
    )
    missing = missing_chunk_columns(present)
    if missing:
        sys.exit(
            "Refusing to ingest. document_chunks is missing: "
            + ", ".join(missing)
            + ". Run 'alembic upgrade head' from backend/. This script does not "
            "create or drop tables."
        )


def main() -> None:
    rows = _load_rows()
    provider = embedding_provider()
    print(f"Loaded {len(rows)} chunks from {CHUNKS_FILE.name}")
    print(f"Embedding provider: {provider} ({EMBEDDING_DIM} dimensions)")
    print("Updating bundled knowledge-base rows only. Student data is not deleted.")

    sources: dict[str, str] = {}
    for row in rows:
        sources.setdefault(row["source"], row.get("title") or row["source"])
    src_to_doc = {source: document_id(source) for source in sources}
    keep_by_doc: dict[str, list[str]] = {doc_id: [] for doc_id in src_to_doc.values()}

    db = get_session()
    try:
        _require_schema(db)
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        db.commit()

        for source, title in sources.items():
            db.execute(text(UPSERT_DOCUMENT_SQL), {"id": src_to_doc[source], "title": title})
        db.commit()
        print(f"Upserted {len(sources)} bundled document rows")

        upsert = text(UPSERT_CHUNK_SQL)
        total = 0
        for start in range(0, len(rows), BATCH_SIZE):
            part = rows[start:start + BATCH_SIZE]
            vectors = embed_texts([row["text"] for row in part], show_progress=False)
            for row, vector in zip(part, vectors):
                if len(vector) != EMBEDDING_DIM:
                    sys.exit(f"Refusing to store a vector of length {len(vector)}; schema is {EMBEDDING_DIM}.")
                cid = chunk_id(row["chunk_id"])
                did = src_to_doc[row["source"]]
                keep_by_doc[did].append(cid)
                db.execute(upsert, {
                    "cid": cid,
                    "did": did,
                    "txt": row["text"],
                    "ttl": row.get("title") or row["source"],
                    "src": row["source"],
                    "pg": row.get("page") or 0,
                    "cat": row.get("category") or "general",
                    "emb": str(vector),
                })
            db.commit()
            total += len(part)
            print(f"  embedded + upserted {total}/{len(rows)}")

        stale = text(STALE_CHUNK_DELETE_SQL).bindparams(bindparam("keep_ids", expanding=True))
        removed = 0
        for doc_id, keep_ids in keep_by_doc.items():
            if not keep_ids:
                continue
            result = db.execute(stale, {"doc_id": doc_id, "keep_ids": keep_ids})
            removed += result.rowcount or 0
        db.commit()
        print(f"Removed {removed} stale bundled chunks")

        try:
            db.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ))
            db.commit()
        except Exception as exc:  # noqa: BLE001 - exact search still works without the index
            db.rollback()
            print(f"Vector index unchanged ({exc.__class__.__name__}). Exact search still works.")

        print(f"DONE: {total} bundled chunks upserted. Student rows were not modified.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
