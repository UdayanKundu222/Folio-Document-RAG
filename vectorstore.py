import io
import os
import re
import shutil
import ssl
import threading

import httpx
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# ── SSL patch for corporate proxy (self-signed cert) ─────────────────────────

if not settings.ssl_verify:
    ssl._create_default_https_context = ssl._create_unverified_context

    _orig_client_init = httpx.Client.__init__

    def _no_verify_client_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _no_verify_client_init

    _orig_async_client_init = httpx.AsyncClient.__init__

    def _no_verify_async_client_init(self, *args, **kwargs):
        kwargs["verify"] = False
        _orig_async_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _no_verify_async_client_init


# ── Singleton embedding model ─────────────────────────────────────────────────
# Loaded once so we don't re-load the weights on every request.

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


# Ingest does load → merge → save. Two concurrent uploads to one collection would
# otherwise both load the old index, and the second save would drop the first's chunks.
_write_lock = threading.Lock()


# ── Path helper ───────────────────────────────────────────────────────────────

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _collection_path(collection_name: str) -> str:
    """Return the on-disk directory for a named FAISS collection.

    The name is validated because it becomes a path: without this, a name like
    "../../important" would let DELETE /collections/... remove arbitrary folders.
    """
    if not _COLLECTION_RE.fullmatch(collection_name):
        raise ValueError("Collection name may only contain letters, digits, '_' and '-' (max 64).")
    return os.path.join(settings.faiss_persist_dir, collection_name)


# ── Loading documents ─────────────────────────────────────────────────────────

def extract_pages(filename: str, data: bytes) -> list[tuple[str, int | None]]:
    """Return [(text, page_number)] for a .txt, .md or .pdf file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    if ext in (".txt", ".md"):
        return [(data.decode("utf-8", errors="replace"), None)]
    raise ValueError("Only .txt, .md and .pdf files are supported.")


# ── Collection helpers ────────────────────────────────────────────────────────

def _load_vectorstore(collection_name: str) -> FAISS:
    """Load an existing FAISS index from disk. Raises FileNotFoundError if absent."""
    path = _collection_path(collection_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Collection '{collection_name}' not found.")
    # FAISS stores its docstore with pickle. That is safe here only because we load
    # indexes this service wrote itself; never load an index file someone uploaded.
    return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)


def ingest_pages(
    pages: list[tuple[str, int | None]],
    source: str,
    collection_name: str = "default",
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> tuple[int, int]:
    """
    Split pages into chunks, embed them, and persist to disk.
    Each chunk keeps its source filename, page and chunk index so answers can cite it.
    If the collection already exists, new chunks are merged in.
    Returns (chunks added in this call, total chunks in the collection).
    """
    path = _collection_path(collection_name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for text, page in pages:
        for piece in splitter.split_text(text):
            meta = {"source": source, "chunk": len(chunks)}
            if page is not None:
                meta["page"] = page
            chunks.append(Document(page_content=piece, metadata=meta))
    if not chunks:
        raise ValueError("No text could be extracted from the file.")

    new_store = FAISS.from_documents(chunks, get_embeddings())
    with _write_lock:
        if os.path.isdir(path):
            store = _load_vectorstore(collection_name)
            store.merge_from(new_store)
        else:
            os.makedirs(path, exist_ok=True)
            store = new_store
        store.save_local(path)
        return len(chunks), store.index.ntotal


def ingest_text(
    text: str,
    collection_name: str = "default",
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
    source: str = "text",
) -> int:
    """Convenience wrapper for plain text. Returns the number of chunks added."""
    added, _ = ingest_pages([(text, None)], source, collection_name, chunk_size, chunk_overlap)
    return added


def similarity_search(
    query: str,
    collection_name: str = "default",
    top_k: int = 4,
) -> list[dict]:
    """
    Return top-k similar chunks for *query* from the named collection.
    Each result is {"content": str, "metadata": dict, "score": float}.
    """
    store = _load_vectorstore(collection_name)
    hits = store.similarity_search_with_score(query, k=top_k)
    return [
        {"content": doc.page_content, "metadata": doc.metadata, "score": float(score)}
        for doc, score in hits
    ]


def get_retriever(collection_name: str = "default", top_k: int = 4):
    """Return a LangChain retriever for use inside a RAG chain."""
    store = _load_vectorstore(collection_name)
    return store.as_retriever(search_kwargs={"k": top_k})


def collection_count(collection_name: str = "default") -> int:
    """Return the number of chunks stored in the named collection."""
    return _load_vectorstore(collection_name).index.ntotal


def list_collections() -> list[str]:
    """Return the names of all collections on disk."""
    root = settings.faiss_persist_dir
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if _COLLECTION_RE.fullmatch(name) and os.path.isdir(os.path.join(root, name))
    )


def delete_collection(collection_name: str = "default") -> None:
    """Delete the named FAISS collection from disk. Raises FileNotFoundError if absent."""
    path = _collection_path(collection_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Collection '{collection_name}' not found.")
    with _write_lock:
        shutil.rmtree(path)
