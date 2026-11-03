"""
RAG LangChain chains — LCEL-based (LangChain >= 0.3.7, no deprecated RetrievalQA).
"""

from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from vectorstore import get_retriever

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You answer questions about the user's documents.
Use only the numbered context passages below, and cite the passages you used like [1] or [2].
Write plain text: no LaTeX, no Markdown headings or tables (the UI shows raw text).
If the context does not contain the answer, reply exactly:
"I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    """Number each passage and label its source so the model can cite it."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        where = meta.get("source", "unknown")
        if "page" in meta:
            where += f", page {meta['page']}"
        parts.append(f"[{i}] ({where})\n{doc.page_content}")
    return "\n\n".join(parts)


# Tried in order if the configured model is over quota (429), overloaded (503/504) or retired (404).
# Flash-Lite answers in ~1 s; the bigger Flash models were taking 10-30 s or timing out.
FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-flash-latest", "gemini-3-flash-preview"]


def _gemini(model: str) -> ChatGoogleGenerativeAI:
    # temperature 0: answers should be reproducible and stick to the context.
    # timeout 15 s + no retries: a busy model fails fast and the next one is tried,
    # instead of the request hanging for minutes. (No thinking_budget: Flash-Lite rejects it.)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
        max_output_tokens=512,
        timeout=15,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_llm():
    fallbacks = [_gemini(m) for m in FALLBACK_MODELS if m != settings.gemini_model]
    return _gemini(settings.gemini_model).with_fallbacks(fallbacks)


def _answer_chain():
    return RAG_PROMPT | get_llm() | StrOutputParser()


def run_rag_qa(
    question: str,
    collection_name: str = "default",
    top_k: int = 4,
) -> dict:
    """
    Retrieve relevant chunks then generate an answer with Gemini via LCEL.

    Returns:
        {
            "answer": str,
            "source_documents": [{"content": str, "metadata": dict}, ...]
        }
    """
    retriever = get_retriever(collection_name=collection_name, top_k=top_k)

    # Retrieve separately so the same chunks are both sent to the LLM and returned as sources
    source_docs = retriever.invoke(question)
    answer = _answer_chain().invoke({"context": _format_docs(source_docs), "question": question})

    return {
        "answer": answer,
        "source_documents": [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in source_docs
        ],
    }


def run_batch_rag_qa(
    questions: list[str],
    collection_name: str = "default",
    top_k: int = 4,
) -> list[dict]:
    """
    Run RAG QA over a list of questions.
    Retrieval runs per question; the LLM calls run concurrently via .batch().
    Returns a list of {"question": str, "answer": str} dicts.
    """
    retriever = get_retriever(collection_name=collection_name, top_k=top_k)
    inputs = [
        {"context": _format_docs(retriever.invoke(q)), "question": q}
        for q in questions
    ]
    answers = _answer_chain().batch(inputs, config={"max_concurrency": 4})
    return [{"question": q, "answer": a} for q, a in zip(questions, answers)]
