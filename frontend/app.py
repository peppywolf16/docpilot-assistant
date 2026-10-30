import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="DocPilot", page_icon="📚", layout="wide")

st.title("📚 DocPilot: Technical Docs Assistant")
st.caption("Ask questions against your documentation with verified citations.")

# --- SIDEBAR: Status & File Uploader ---
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            data = health_resp.json()
            st.success(f"Backend Online ({data.get('total_chunks', 0)} chunks indexed)")
        else:
            st.error("Backend Error")
    except requests.exceptions.ConnectionError:
        st.error("Backend Offline! Start uvicorn on port 8000.")

    st.divider()

    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .md or .txt files",
        type=["md", "txt"],
        accept_multiple_files=True
    )

    if st.button("Index Uploaded Files", disabled=not uploaded_files):
        with st.spinner("Ingesting and embedding files..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "text/plain"))
                for f in uploaded_files
            ]
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", files=files_payload)
                if resp.status_code == 200:
                    res_data = resp.json()
                    st.success(f"Indexed {res_data['chunks_added']} new chunks!")
                    st.rerun()
                else:
                    st.error(f"Failed to ingest: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    top_k = st.slider("Top-K Passages to Retrieve", min_value=1, max_value=8, value=4)


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['number']}] {s['doc']}** (Chunk ID: `{s['chunk_id']}`, Relevance: `{s['score']}`)")
                    st.code(s["text"], language="markdown")

# Handle user input
if user_query := st.chat_input("Ask a question about your docs..."):
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Searching docs and generating answer..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": user_query, "top_k": top_k},
                    timeout=30
                )
                if resp.status_code == 200:
                    result = resp.json()
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if sources:
                        with st.expander("🔍 View Retrieved Sources"):
                            for s in sources:
                                st.markdown(f"**[{s['number']}] {s['doc']}** (Chunk ID: `{s['chunk_id']}`, Relevance: `{s['score']}`)")
                                st.code(s["text"], language="markdown")

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")