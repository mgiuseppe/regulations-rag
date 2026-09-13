## Flow
Query (from the Streamlit chat):
   - Embed the user's question
   - Semantic search on Qdrant, retrieve top-N chunks
   - Pass the retrieved chunks + user's question to gpt-5-nano
   - Show the answer, plus one expandable citation per chunk used