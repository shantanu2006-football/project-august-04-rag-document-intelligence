"""Generate the one sample PDF document in data/corpus/.

The corpus is meant to include both .txt and .pdf sources so the loader's
PDF-extraction path is exercised by something other than a test fixture. The
source text lives in this script so the PDF is reproducible; the script is a
one-off content-authoring tool, not something the runtime package imports.

Usage:
    python scripts/generate_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

TITLE = "Retrieval-Augmented Generation: An Overview"

PARAGRAPHS = [
    "Retrieval-augmented generation (RAG) is a pattern for grounding a language "
    "model's answers in a specific corpus of documents rather than relying solely "
    "on what the model memorized during training. Instead of asking the model to "
    "answer a question directly, a RAG system first retrieves the passages most "
    "relevant to the question from an external index, then asks the model to "
    "answer using only that retrieved context.",
    "The main motivation is factual grounding. A language model's parametric "
    "knowledge is frozen at training time, can be wrong, and cannot be traced "
    "back to a source. Retrieval lets a system answer questions about documents "
    "it has never been trained on, keep its knowledge current by simply updating "
    "the index, and cite the exact passage an answer came from, which lets a "
    "user verify the answer rather than trust it blindly.",
    "A RAG pipeline has three stages that each affect answer quality "
    "independently. Chunking splits source documents into passages small enough "
    "to embed meaningfully and to fit in the model's context window, but large "
    "enough to preserve the surrounding context a sentence needs to be "
    "understood; a common strategy is a fixed token or character budget per "
    "chunk with a small overlap between consecutive chunks so a fact near a "
    "chunk boundary is not split away from the sentence that explains it.",
    "Retrieval embeds the user's question and the corpus chunks into the same "
    "vector space and returns the chunks whose embeddings are closest to the "
    "question's embedding, usually by cosine similarity. Retrieval quality is "
    "measured independently of generation quality, typically with precision and "
    "recall at a cutoff k against a hand-labeled set of question and "
    "relevant-chunk pairs, since a wrong or missing retrieval dooms the final "
    "answer regardless of how good the generation step is.",
    "Generation (or, in a purely extractive system, answer selection) conditions "
    "on the retrieved chunks and the original question to produce a final "
    "answer. Even a strong generator cannot recover from a retrieval step that "
    "missed the relevant passage, which is why production RAG systems invest "
    "heavily in retrieval evaluation and treat it as the primary lever for "
    "improving end-to-end answer quality, well before tuning the generation "
    "step itself.",
]


def build_pdf(output_path: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, TITLE)
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    for paragraph in PARAGRAPHS:
        pdf.multi_cell(0, 7, paragraph)
        pdf.ln(4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    build_pdf(repo_root / "data" / "corpus" / "rag_systems.pdf")
    print("Wrote data/corpus/rag_systems.pdf")
