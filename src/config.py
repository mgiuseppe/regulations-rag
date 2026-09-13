from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class EmbeddingModelConfig:
    name: str
    cost_per_1m_tokens: float
    max_tokens: int
    tokenizer_name: str


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    "text-embedding-3-small": EmbeddingModelConfig(
        name="text-embedding-3-small",
        cost_per_1m_tokens=0.02,
        max_tokens=8191,
        tokenizer_name="cl100k_base",
    ),
    "text-embedding-3-large": EmbeddingModelConfig(
        name="text-embedding-3-large",
        cost_per_1m_tokens=0.13,
        max_tokens=8191,
        tokenizer_name="cl100k_base",
    ),
}

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "regulations"
VECTOR_SIZE = 1536

GENERATION_MODEL = "gpt-5-nano"
TOP_N_CHUNKS = 5
