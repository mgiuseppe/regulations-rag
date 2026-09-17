import streamlit as st
import logging
import time

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
            with st.expander(chunk.citation):
                st.write(chunk.text)


for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask a question about the ingested regulations...")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "chunks": None})
    render_message(st.session_state.messages[-1])

    retrieval_service = RetrievalService()

    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]

    answer_start = time.perf_counter()
    result = retrieval_service.answer(question, history=history)

    with st.chat_message("assistant"):
        answer = st.write_stream(result.text)
        logger.info("chat page: answer took %.3fs", time.perf_counter() - answer_start)
        for chunk in result.citation_chunks:
            with st.expander(chunk.citation):
                st.write(chunk.text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": result.citation_chunks}
    )

logger.info(
    "chat page: total %.3fs",
    time.perf_counter() - page_start
)
