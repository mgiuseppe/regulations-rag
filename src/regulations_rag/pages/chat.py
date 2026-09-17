import streamlit as st
import logging
import time

from regulations_rag.services import retrieval
from regulations_rag.services.retrieval import Message, RetrievalService

st.title("Chat")

logger = logging.getLogger(__name__)
page_start = time.perf_counter()
logger.info("chat page: start")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "chunks": list[RetrievedChunk] | None}


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for chunk in message.get("chunks") or []:
            with st.expander(retrieval.format_citation(chunk)):
                st.write(chunk.text)


for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask a question about the ingested regulations...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "chunks": None})
    render_message(st.session_state.messages[-1])

    retrieval_service = RetrievalService()

    embedding_start = time.perf_counter()
    query_vector = retrieval_service.embed_query(question)
    logger.info("chat page: embedding query took %.3fs", time.perf_counter() - embedding_start)

    semantic_search_start = time.perf_counter()
    chunks = retrieval_service.search(query_vector)
    logger.info("chat page: semantic search took %.3fs", time.perf_counter() - semantic_search_start)

    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]
    messages = retrieval_service.build_messages(question, chunks, history)

    with st.chat_message("assistant"):
        llm_answer_start = time.perf_counter()
        answer = st.write_stream(retrieval_service.generate_answer(messages))
        logger.info("chat page: llm answer took %.3fs", time.perf_counter() - llm_answer_start)
        for chunk in chunks:
            with st.expander(retrieval.format_citation(chunk)):
                st.write(chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})

logger.info(
    "chat page: total %.3fs",
    time.perf_counter() - page_start
)
