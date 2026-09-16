from dataclasses import dataclass
from typing import Iterator, Literal

from openai import OpenAI

from regulations_rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODELS, GENERATION_MODEL
from regulations_rag.repositories.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a regulatory assistant. Answer the user's question using ONLY the "
    "context chunks provided below, which are excerpts from regulations. "
    "Never use any knowledge you have outside of these chunks. "
    "If the provided context does not contain enough information to answer the "
    "question, say so explicitly instead of guessing or making anything up."
)

Role = Literal["user", "assistant"]


@dataclass
class RetrievedChunk:
    title: str
    source_file: str
    page: int
    text: str
    score: float


@dataclass
class Message:
    role: Role
    content: str


def embed_query(question: str, openai_client: OpenAI, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    embedding_config = EMBEDDING_MODELS[model_name]
    response = openai_client.embeddings.create(model=embedding_config.name, input=[question])
    return response.data[0].embedding


def search(
    vector_store: VectorStore, query_vector: list[float], top_n: int
) -> list[RetrievedChunk]:
    results = vector_store.search(query_vector, top_n)
    return [
        RetrievedChunk(
            title=result.title,
            source_file=result.source_file,
            page=result.page,
            text=result.text,
            score=result.score,
        )
        for result in results
    ]


def format_citation(chunk: RetrievedChunk) -> str:
    return f"{chunk.title}, page {chunk.page}"


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context was found in the regulations database."
    return "\n\n".join(
        f"[{i}] {format_citation(chunk)}\n{chunk.text}" for i, chunk in enumerate(chunks, start=1)
    )


def build_messages(
    question: str, chunks: list[RetrievedChunk], history: list[Message]
) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}",
        }
    )
    return messages


def generate_answer(
    messages: list[dict], openai_client: OpenAI, model: str = GENERATION_MODEL
) -> Iterator[str]:
    stream = openai_client.chat.completions.create(model=model, messages=messages, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
