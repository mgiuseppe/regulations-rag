import streamlit as st
from openai import OpenAI

from regulations_rag import retrieval, vector_store
from regulations_rag.config import TOP_N_CHUNKS
from regulations_rag.retrieval import Message

st.title("Chat")

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

    openai_client = OpenAI()
    qdrant_client = vector_store.get_client()

    query_vector = retrieval.embed_query(question, openai_client)
    chunks = retrieval.search(qdrant_client, query_vector, TOP_N_CHUNKS)

    history = [
        Message(role=m["role"], content=m["content"])
        for m in st.session_state.messages[:-1]
    ]
    messages = retrieval.build_messages(question, chunks, history)

    with st.chat_message("assistant"):
        answer = st.write_stream(retrieval.generate_answer(messages, openai_client))
        for chunk in chunks:
            with st.expander(retrieval.format_citation(chunk)):
                st.write(chunk.text)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
