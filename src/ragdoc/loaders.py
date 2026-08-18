"""Load raw documents (.txt, .md, .pdf) from a directory into plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class Document:
    """A single source document before chunking."""

    doc_id: str
    source_path: str
    text: str


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_document(path: Path) -> Document:
    """Load a single supported file into a Document.

    Raises:
        ValueError: if the file extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf_file(path)
    elif suffix in {".txt", ".md"}:
        text = _load_text_file(path)
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path}; "
            f"supported types are {sorted(SUPPORTED_SUFFIXES)}"
        )
    text = text.strip()
    return Document(doc_id=path.stem, source_path=str(path), text=text)


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every supported file in corpus_dir into a list of Documents.

    Files are returned sorted by doc_id for deterministic ordering. Empty
    documents (e.g. a PDF with no extractable text) are skipped with no error,
    since an empty document contributes no chunks and no retrievable content.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    paths = sorted(
        p for p in corpus_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = []
    for path in paths:
        doc = load_document(path)
        if doc.text:
            documents.append(doc)
    if not documents:
        raise ValueError(f"No loadable documents found in {corpus_dir}")
    return documents
