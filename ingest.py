"""Ingest one complete resume PDF into one persistent Chroma document."""

import argparse
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LEGACY_DATA_DIR = PROJECT_ROOT / "resume_dataset"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "resume_screening"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_data_dir() -> Path:
    """Use data/ when present, while accepting the supplied initial folder."""
    return DATA_DIR if DATA_DIR.exists() else LEGACY_DATA_DIR


def metadata_for(pdf_path: Path) -> dict[str, str]:
    """Build candidate metadata without depending on hard-coded resume names."""
    match = re.match(r"Resume_([^_]+)_(.+)$", pdf_path.stem, re.IGNORECASE)
    if match:
        resume_id, name_part = match.groups()
        candidate = name_part.replace("_", " ")
    else:
        resume_id = pdf_path.stem
        candidate = pdf_path.stem.replace("_", " ")

    return {
        "resume_id": resume_id,
        "candidate": candidate,
        "source": pdf_path.name,
    }


def load_resume(pdf_path: Path):
    """Load every page, then combine those pages into one LangChain Document."""
    pages = PyPDFLoader(str(pdf_path)).load()
    complete_text = "\n\n".join(page.page_content for page in pages)
    metadata = metadata_for(pdf_path)

    # One PDF is intentionally one Document: no text splitter and no page documents.
    from langchain_core.documents import Document

    return Document(page_content=complete_text, metadata=metadata)


def build_vector_store(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Chroma:
    """Create the persistent Chroma collection with the same embedding model used by rag.py."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def ingest(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Scan all PDFs and upsert one embedding per resume, keyed by filename stem."""
    data_dir = get_data_dir()
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF resumes found in {data_dir}")

    documents = [load_resume(pdf_path) for pdf_path in pdf_paths]
    ids = [pdf_path.stem for pdf_path in pdf_paths]
    vector_store = build_vector_store(embedding_model)

    # Stable IDs make rerunning ingestion update the same resume instead of duplicating it.
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Ingested {len(documents)} complete resumes from {data_dir}")
    print(f"ChromaDB directory: {CHROMA_DIR}")
    return len(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest all resume PDFs into ChromaDB")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    ingest(args.embedding_model)

