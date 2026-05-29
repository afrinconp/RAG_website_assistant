import os
import uuid

import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics.metrics import load_messages_df, top_user_terms
from app.config.settings import get_settings
from app.memory.service import ConversationService
from app.rag.processing import process_documents
from app.rag.vector_store import ChromaVectorStore
from app.scraper.pipeline import run_scraping_pipeline


st.set_page_config(page_title="RAG Bank Assistant", layout="wide")
st.title("RAG Bank Assistant")
st.caption("Web scraping + semantic retrieval + Gemini/LangChain RAG + conversation memory + analytics")

settings = get_settings()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("Session Memory")
    session_input = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Use the same Session ID to recover previous conversation context.",
    )
    st.session_state.session_id = session_input.strip() or st.session_state.session_id

    c_new, c_clear = st.columns(2)
    if c_new.button("New session"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    service_for_sidebar = ConversationService()
    if c_clear.button("Clear session"):
        service_for_sidebar.clear_session(st.session_state.session_id)
        st.success("Session memory cleared.")

    st.caption(f"Current Session ID: `{st.session_state.session_id}`")
    st.write(f"Start URL: {settings.start_url}")

    st.divider()
    st.header("Pipeline")
    if st.button("1. Run scraper"):
        with st.spinner("Scraping website..."):
            count = run_scraping_pipeline()
        st.success(f"Scraped {count} pages.")

    if st.button("2. Process data"):
        with st.spinner("Cleaning and chunking..."):
            count = process_documents()
        st.success(f"Created {count} chunks.")

    if st.button("3. Build vector index"):
        with st.spinner("Embedding and indexing chunks..."):
            count = ChromaVectorStore().build_index()
        st.success(f"Indexed {count} chunks.")

    st.divider()
    st.subheader("Config")
    st.write(f"History messages: {settings.n_history_messages}")
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"Reranker enabled: `{settings.use_reranker}`")
    st.write(f"Gemini model: `{settings.gemini_model}`")

    st.divider()
    st.subheader("Gemini API Key")

    current_key = st.session_state.get("google_api_key", os.getenv("GOOGLE_API_KEY", ""))

    api_key_input = st.text_input(
        "Introduce your Gemini API key",
        value=current_key,
        type="password",
        placeholder="AIza...",
        help="This key is used only during the current Streamlit session.",
    )

    if st.button("Save Gemini API key"):
        if api_key_input.strip():
            st.session_state.google_api_key = api_key_input.strip()
            os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            st.success("Gemini API key saved for this session.")
        else:
            st.warning("Please enter a valid Gemini API key.")

    st.markdown(
        "[Get or create your Gemini API key in Google AI Studio](https://aistudio.google.com/api-keys)"
    )

    if st.session_state.get("google_api_key") or os.getenv("GOOGLE_API_KEY"):
        st.caption("Gemini API key is configured.")
    else:
        st.caption("Gemini API key is not configured yet.")

chat_tab, analytics_tab = st.tabs(["Chat", "Analytics"])

with chat_tab:
    service = ConversationService()
    st.subheader("Ask a question about the indexed website")

    session_messages = service.get_session_messages(st.session_state.session_id)
    if session_messages:
        with st.expander("Ver memoria de conversación de esta sesión", expanded=False):
            for message in session_messages:
                role = "Usuario" if message["role"] == "user" else "Asistente"
                st.markdown(f"**{role}** · {message.get('created_at', '')}")
                st.write(message["content"])
                st.divider()
    else:
        st.info("This session has no previous messages yet. The system will start building memory after your first question.")

    question = st.chat_input("Example: ¿Qué productos ofrece el banco?")
    if question:
        st.chat_message("user").write(question)
        with st.spinner("Retrieving the most similar chunks and generating an answer with Gemini..."):
            result = service.ask(st.session_state.session_id, question)

        st.chat_message("assistant").write(result["answer"])
        st.caption(f"Memory used: {result.get('used_history_messages', 0)} previous messages from this session.")

        with st.expander("Ver consulta contextual usada para retrieval"):
            st.code(result.get("contextual_retrieval_query", question), language="text")

        with st.expander("Ver fuentes recuperadas"):
            for source in result["sources"]:
                st.markdown(f"**Fuente {source['rank']}: {source['title']}**")
                if source.get("url"):
                    st.write(source["url"])
                st.caption(f"Chunk: {source.get('chunk_index')} | Distance: {source.get('distance')} | Similarity: {source.get('similarity_score')} | Rerank score: {source.get('rerank_score')}")
                if source.get("rerank_error"):
                    st.warning(f"Reranker disabled automatically: {source.get('rerank_error')}")
                st.write(source.get("text_preview", ""))
                st.divider()

with analytics_tab:
    st.subheader("Conversation analytics")
    df = load_messages_df()
    if df.empty:
        st.info("No conversation history yet. Ask a question first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total messages", len(df))
        c2.metric("Sessions", df["session_id"].nunique())
        c3.metric("Avg. message length", round(df["message_length"].mean(), 1))

        role_counts = df.groupby("role").size().reset_index(name="count")
        st.plotly_chart(px.bar(role_counts, x="role", y="count", title="Messages by role"), use_container_width=True)

        daily = df.groupby("date").size().reset_index(name="messages")
        st.plotly_chart(px.line(daily, x="date", y="messages", title="Messages over time"), use_container_width=True)

        terms = pd.DataFrame(top_user_terms(df), columns=["term", "count"])
        if not terms.empty:
            st.plotly_chart(px.bar(terms, x="term", y="count", title="Most common user terms"), use_container_width=True)

        st.dataframe(df[["session_id", "role", "content", "created_at"]], use_container_width=True)
