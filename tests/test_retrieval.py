from types import SimpleNamespace

from regulations_rag import retrieval
from regulations_rag.repositories.vector_store import SearchResult
from regulations_rag.retrieval import Message, RetrievedChunk


def make_chunk(title="PPWR", page=3, text="Article 5 says...", score=0.9):
    return RetrievedChunk(title=title, source_file="ppwr.pdf", page=page, text=text, score=score)


def test_format_citation():
    chunk = make_chunk(title="PPWR", page=3)
    assert retrieval.format_citation(chunk) == "PPWR, page 3"


def test_build_messages_includes_system_prompt_and_context():
    chunk = make_chunk()
    messages = retrieval.build_messages("What does article 5 say?", [chunk], [])

    assert messages[0]["role"] == "system"
    assert "ONLY the context chunks" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "PPWR, page 3" in messages[-1]["content"]
    assert "Article 5 says..." in messages[-1]["content"]
    assert "What does article 5 say?" in messages[-1]["content"]


def test_build_messages_with_no_chunks_notes_missing_context():
    messages = retrieval.build_messages("Unanswerable question", [], [])
    assert "No relevant context was found" in messages[-1]["content"]


def test_build_messages_includes_prior_history():
    history = [
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
    ]
    messages = retrieval.build_messages("Second question", [], history)

    assert messages[1] == {"role": "user", "content": "First question"}
    assert messages[2] == {"role": "assistant", "content": "First answer"}
    assert messages[-1]["role"] == "user"


def test_search_maps_scored_points_to_retrieved_chunks(mocker):
    fake_result = SearchResult(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    fake_vector_store = mocker.Mock()
    fake_vector_store.search.return_value = [fake_result]

    chunks = retrieval.search(vector_store=fake_vector_store, query_vector=[0.1], top_n=5)

    assert chunks == [
        RetrievedChunk(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    ]


def test_embed_query_calls_embeddings_api():
    captured = {}

    def fake_create(model, input):
        captured["model"] = model
        captured["input"] = input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

    vector = retrieval.embed_query("hello", fake_client)

    assert vector == [0.1, 0.2]
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["hello"]


def test_generate_answer_yields_content_deltas():
    def fake_event(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    fake_stream = [fake_event("Hel"), fake_event("lo"), fake_event(None)]
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: iter(fake_stream))
        )
    )

    result = "".join(retrieval.generate_answer([], fake_client))

    assert result == "Hello"
