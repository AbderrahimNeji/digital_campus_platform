"""Ids and SQL for the bundled faculty knowledge base.

Document and chunk ids are UUID5 values in a fixed namespace, so running
ingestion again updates the same rows. The only delete is of stale global
chunks that belong to those bundled documents. Student accounts, courses,
course materials, chat, and student chunks are never mentioned.
"""

from __future__ import annotations

import uuid

NS = uuid.UUID("9b1d2c3e-0000-4a00-8a00-abcdef010001")

REQUIRED_CHUNK_COLUMNS = frozenset({
    "chunk_id",
    "document_id",
    "text",
    "title",
    "source",
    "page",
    "category",
    "embedding",
    "student_id",
    "material_id",
})

UPSERT_DOCUMENT_SQL = """
INSERT INTO documents (id, title, content)
VALUES (:id, :title, '')
ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title
"""

UPSERT_CHUNK_SQL = """
INSERT INTO document_chunks
    (chunk_id, document_id, text, title, source, page, category, embedding,
     student_id, course_id, material_id)
VALUES
    (:cid, :did, :txt, :ttl, :src, :pg, :cat, CAST(:emb AS vector),
     NULL, NULL, NULL)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    text = EXCLUDED.text,
    title = EXCLUDED.title,
    source = EXCLUDED.source,
    page = EXCLUDED.page,
    category = EXCLUDED.category,
    embedding = EXCLUDED.embedding
WHERE document_chunks.student_id IS NULL
  AND document_chunks.material_id IS NULL
"""

STALE_CHUNK_DELETE_SQL = """
DELETE FROM document_chunks
WHERE student_id IS NULL
  AND material_id IS NULL
  AND document_id = CAST(:doc_id AS uuid)
  AND chunk_id NOT IN :keep_ids
"""


def document_id(source: str) -> uuid.UUID:
    return uuid.uuid5(NS, "doc:" + source)


def chunk_id(raw_chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(NS, raw_chunk_id)


def missing_chunk_columns(present: set[str]) -> list[str]:
    return sorted(REQUIRED_CHUNK_COLUMNS - present)
