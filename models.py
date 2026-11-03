from typing import Any

from pydantic import BaseModel, Field

# Collection names become folder names on disk, so only allow a safe character set.
COLLECTION_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


# ── Request Models ───────────────────────────────────────────────────────────

class SimilaritySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    collection_name: str = Field(default="default", pattern=COLLECTION_PATTERN)
    top_k: int = Field(default=4, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    collection_name: str = Field(default="default", pattern=COLLECTION_PATTERN)
    top_k: int = Field(default=4, ge=1, le=20)


class BatchAskRequest(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=20)
    collection_name: str = Field(default="default", pattern=COLLECTION_PATTERN)
    top_k: int = Field(default=4, ge=1, le=20)


# ── Response Models ──────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # L2 distance from the query embedding: lower means more similar
    score: float | None = None


class SimilaritySearchResponse(BaseModel):
    results: list[SourceDocument]


class AskResponse(BaseModel):
    answer: str
    source_documents: list[SourceDocument]


class BatchAskItem(BaseModel):
    question: str
    answer: str


class BatchAskResponse(BaseModel):
    results: list[BatchAskItem]


class IngestResponse(BaseModel):
    collection_name: str
    filename: str
    chunks_added: int
    total_chunks: int


class CollectionInfoResponse(BaseModel):
    collection_name: str
    count: int


class CollectionListResponse(BaseModel):
    collections: list[CollectionInfoResponse]


class DeleteResponse(BaseModel):
    collection_name: str
    deleted: bool
