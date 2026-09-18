from dataclasses import dataclass
from pathlib import Path

import tiktoken
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from regulations_rag.config import EmbeddingModelConfig

# Below this many characters of extracted text per page, we assume the PDF
# is scanned/image-based and re-parse with OCR enabled.
MIN_CHARS_PER_PAGE_WITHOUT_OCR = 20


@dataclass
class Chunk:
    page: int
    text: str


def _convert(path: Path, do_ocr: bool):
    pipeline_options = PdfPipelineOptions(do_ocr=do_ocr)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    return converter.convert(str(path)).document


def _needs_ocr(document) -> bool:
    num_pages = document.num_pages()
    if num_pages == 0:
        return False
    total_chars = len(document.export_to_text())
    return (total_chars / num_pages) < MIN_CHARS_PER_PAGE_WITHOUT_OCR


def parse_and_chunk(path: Path, embedding_config: EmbeddingModelConfig) -> list[Chunk]:
    document = _convert(path, do_ocr=False)
    if _needs_ocr(document):
        document = _convert(path, do_ocr=True)

    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.get_encoding(embedding_config.tokenizer_name),
        max_tokens=embedding_config.max_tokens,
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    chunks = []
    for chunk in chunker.chunk(document):
        page = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else 1
        chunks.append(Chunk(page=page, text=chunker.contextualize(chunk)))
    return chunks
