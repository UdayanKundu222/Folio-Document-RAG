from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from chains import get_llm, run_batch_rag_qa, run_rag_qa
from config import settings
from models import (
    AskRequest,
    AskResponse,
    BatchAskItem,
    BatchAskRequest,
    BatchAskResponse,
    CollectionInfoResponse,
    CollectionListResponse,
    DeleteResponse,
    IngestResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    SourceDocument,
)
from vectorstore import (
    collection_count,
    delete_collection,
    extract_pages,
    get_embeddings,
    ingest_pages,
    list_collections,
    similarity_search,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Warm up at startup: loading the embedding model takes ~20 s the first time,
    # which would otherwise land on whoever asks the first question.
    get_embeddings().embed_query("warm-up")
    get_llm()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="RAG Document QnA API",
    description=(
        "Retrieval-Augmented Generation API backed by FAISS, "
        "HuggingFace embeddings and Google Gemini. Upload .txt, .md or .pdf files "
        "into a collection, then search it or ask questions with cited sources."
    ),
    version="2.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"


def _not_found(collection_name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found.")


# ── UI & Health ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Health"])
def health():
    """Returns 200 OK when the service is running."""
    return {"status": "ok"}


# ── Collections ──────────────────────────────────────────────────────────────

@app.get("/collections", response_model=CollectionListResponse, tags=["Vector Store"])
def collections():
    """List every collection and its chunk count."""
    return CollectionListResponse(
        collections=[
            CollectionInfoResponse(collection_name=name, count=collection_count(name))
            for name in list_collections()
        ]
    )


@app.post(
    "/collections/{collection_name}/documents",
    response_model=IngestResponse,
    status_code=201,
    tags=["Vector Store"],
)
def upload_document(collection_name: str, file: UploadFile = File(...)):
    """
    Upload a .txt, .md or .pdf file. It is split into overlapping chunks,
    embedded and added to the collection (created if it doesn't exist).
    """
    limit = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB.")
    filename = file.filename or "upload.txt"
    try:
        pages = extract_pages(filename, data)
        added, total = ingest_pages(pages, source=filename, collection_name=collection_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return IngestResponse(
        collection_name=collection_name,
        filename=filename,
        chunks_added=added,
        total_chunks=total,
    )


@app.get(
    "/collections/{collection_name}",
    response_model=CollectionInfoResponse,
    tags=["Vector Store"],
)
def collection_info(collection_name: str):
    """Return the number of chunks stored in the named FAISS collection."""
    try:
        count = collection_count(collection_name)
    except FileNotFoundError:
        raise _not_found(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return CollectionInfoResponse(collection_name=collection_name, count=count)


@app.delete(
    "/collections/{collection_name}",
    response_model=DeleteResponse,
    tags=["Vector Store"],
)
def delete(collection_name: str):
    """Permanently delete a FAISS collection folder and all its embeddings."""
    try:
        delete_collection(collection_name)
    except FileNotFoundError:
        raise _not_found(collection_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return DeleteResponse(collection_name=collection_name, deleted=True)


# ── Similarity Search ─────────────────────────────────────────────────────────

@app.post(
    "/similarity-search",
    response_model=SimilaritySearchResponse,
    tags=["Vector Store"],
)
def semantic_search(request: SimilaritySearchRequest):
    """
    Return the top-k most similar chunks to the query **without** calling the LLM,
    with their distance scores. Useful for debugging retrieval quality.
    """
    try:
        raw = similarity_search(request.query, request.collection_name, request.top_k)
    except FileNotFoundError:
        raise _not_found(request.collection_name)
    return SimilaritySearchResponse(results=[SourceDocument(**r) for r in raw])


# ── RAG QnA ──────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["RAG QnA"])
def ask(request: AskRequest):
    """
    Answer a question using Retrieval-Augmented Generation:

    1. Embed the question and retrieve the top-k relevant chunks from FAISS.
    2. Pass the numbered chunks + question to Gemini via a LangChain LCEL pipeline.
    3. Return the answer (with [n] citations) and the source chunks used.
    """
    try:
        result = run_rag_qa(request.question, request.collection_name, request.top_k)
    except FileNotFoundError:
        raise _not_found(request.collection_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return AskResponse(
        answer=result["answer"],
        source_documents=[SourceDocument(**d) for d in result["source_documents"]],
    )


@app.post("/ask-batch", response_model=BatchAskResponse, tags=["RAG QnA"])
def ask_batch(request: BatchAskRequest):
    """Answer up to 20 questions in one request using the same RAG pipeline."""
    try:
        results = run_batch_rag_qa(request.questions, request.collection_name, request.top_k)
    except FileNotFoundError:
        raise _not_found(request.collection_name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return BatchAskResponse(results=[BatchAskItem(**r) for r in results])
