from types import SimpleNamespace

from regulations_rag.repositories.vector_store import SearchResult
from regulations_rag.services.retrieval import Message, RetrievalService, RetrievedChunk


def make_chunk(title="PPWR", page=3, text="Article 5 says...", score=0.9):
    return RetrievedChunk(title=title, source_file="ppwr.pdf", page=page, text=text, score=score)


def make_service(llm_client=None, vector_store=None):
    return RetrievalService(llm_client=llm_client, vector_store=vector_store)


def test_retrieved_chunk_citation():
    chunk = make_chunk(title="PPWR", page=3)
    assert chunk.citation == "PPWR, page 3"


def test_build_messages_includes_system_prompt_and_context():
    chunk = make_chunk()
    service = make_service()
    messages = service._build_messages("What does article 5 say?", [chunk], [])

    assert messages[0]["role"] == "system"
    assert "ONLY the context chunks" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert "PPWR, page 3" in messages[-1]["content"]
    assert "Article 5 says..." in messages[-1]["content"]
    assert "What does article 5 say?" in messages[-1]["content"]


def test_build_messages_with_no_chunks_notes_missing_context():
    service = make_service()
    messages = service._build_messages("Unanswerable question", [], [])
    assert "No relevant context was found" in messages[-1]["content"]


def test_build_messages_includes_prior_history():
    history = [
        Message(role="user", content="First question"),
        Message(role="assistant", content="First answer"),
    ]
    service = make_service()
    messages = service._build_messages("Second question", [], history)

    assert messages[1] == {"role": "user", "content": "First question"}
    assert messages[2] == {"role": "assistant", "content": "First answer"}
    assert messages[-1]["role"] == "user"


def test_search_maps_scored_points_to_retrieved_chunks(mocker):
    fake_result = SearchResult(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    fake_vector_store = mocker.Mock()
    fake_vector_store.search.return_value = [fake_result]

    service = make_service(vector_store=fake_vector_store)
    chunks = service._search(query_vector=[0.1], top_n=5)

    assert chunks == [
        RetrievedChunk(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    ]


def test_search_defaults_top_n_to_config_value(mocker):
    from regulations_rag.config import TOP_N_CHUNKS

    fake_vector_store = mocker.Mock()
    fake_vector_store.search.return_value = []

    service = make_service(vector_store=fake_vector_store)
    service._search(query_vector=[0.1])

    fake_vector_store.search.assert_called_once_with([0.1], TOP_N_CHUNKS)


def test_embed_query_calls_embeddings_api():
    captured = {}

    def fake_create(model, input):
        captured["model"] = model
        captured["input"] = input
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=fake_create))

    service = make_service(llm_client=fake_client)
    vector = service._embed_query("hello")

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

    service = make_service(llm_client=fake_client)
    result = "".join(service._generate_answer([]))

    assert result == "Hello"


def test_answer_embeds_searches_and_streams_generation(mocker):
    fake_result = SearchResult(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    fake_vector_store = mocker.Mock()
    fake_vector_store.search.return_value = [fake_result]

    def fake_embeddings_create(model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    def fake_event(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    fake_chat_create = mocker.Mock(return_value=iter([fake_event("Hi"), fake_event(None)]))
    fake_llm_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=fake_embeddings_create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_chat_create)),
    )

    service = make_service(llm_client=fake_llm_client, vector_store=fake_vector_store)
    result = service.answer("What does article 5 say?")

    assert result.citation_chunks == [
        RetrievedChunk(title="PPWR", source_file="ppwr.pdf", page=3, text="hello", score=0.87)
    ]
    assert "".join(result.text) == "Hi"
    fake_vector_store.search.assert_called_once()
    fake_chat_create.assert_called_once()
    called_messages = fake_chat_create.call_args.kwargs["messages"]
    assert "What does article 5 say?" in called_messages[-1]["content"]


def test_answer_includes_history_and_respects_overrides(mocker):
    fake_vector_store = mocker.Mock()
    fake_vector_store.search.return_value = []

    def fake_embeddings_create(model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])

    fake_chat_create = mocker.Mock(return_value=iter([]))
    fake_llm_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=fake_embeddings_create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_chat_create)),
    )

    service = make_service(llm_client=fake_llm_client, vector_store=fake_vector_store)
    history = [Message(role="user", content="Earlier question")]

    result = service.answer(
        "Second question",
        history=history,
        top_n_rank=3,
        llm_model="gpt-test",
    )
    list(result.text)

    fake_vector_store.search.assert_called_once_with([0.1], 3)
    called_messages = fake_chat_create.call_args.kwargs["messages"]
    assert {"role": "user", "content": "Earlier question"} in called_messages
    assert fake_chat_create.call_args.kwargs["model"] == "gpt-test"
