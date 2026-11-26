import os
import tempfile
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CACHED_THREADS,
    OLLAMA_EMBED_MODEL,
    RETRIEVER_TOP_K,
)

embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def _get_stores() -> Dict[str, Any]:
    return {
        "vectors": OrderedDict(),   # thread_id -> FAISS store (supports several PDFs per thread)
        "metadata": OrderedDict(),
        "lock": threading.Lock(),
    }


# Initialize on main Streamlit thread so LangGraph worker threads never hit NoSessionContext
_STORES = _get_stores()


def _evict_if_needed(current: str):
    while current not in _STORES["vectors"] and len(_STORES["vectors"]) >= MAX_CACHED_THREADS:
        oldest_key, _ = _STORES["vectors"].popitem(last=False)
        _STORES["metadata"].pop(oldest_key, None)


def get_retriever(thread_id: Optional[str]) -> Optional[Any]:
    if not thread_id:
        return None
    with _STORES["lock"]:
        vs = _STORES["vectors"].get(str(thread_id))
    if vs is None:
        return None
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_TOP_K})


def thread_has_document(thread_id: str) -> bool:
    with _STORES["lock"]:
        return str(thread_id) in _STORES["vectors"]


def thread_document_metadata(thread_id: str) -> dict:
    with _STORES["lock"]:
        return dict(_STORES["metadata"].get(str(thread_id), {}))


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    thread_id = str(thread_id)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        display_name = filename or os.path.basename(tmp_path)
        for d in docs:
            d.metadata["source"] = display_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
        if not chunks:
            raise ValueError("No extractable text found (scanned PDF? OCR is needed).")

        with _STORES["lock"]:
            existing = _STORES["vectors"].get(thread_id)

        if existing is not None:
            existing.add_documents(chunks)          # keep previously uploaded PDFs
            vs = existing
        else:
            vs = FAISS.from_documents(chunks, embeddings)

        with _STORES["lock"]:
            _evict_if_needed(thread_id)
            old = _STORES["metadata"].get(thread_id, {})
            files = old.get("files", []) + [display_name]
            meta = {
                "files": files,
                "filename": ", ".join(files),
                "documents": old.get("documents", 0) + len(docs),
                "chunks": old.get("chunks", 0) + len(chunks),
            }
            _STORES["vectors"][thread_id] = vs
            _STORES["metadata"][thread_id] = meta
        return meta
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
