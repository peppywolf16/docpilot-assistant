import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="DocPilot v0.2", page_icon="📚", layout="wide")

st.title("📚 DocPilot: Technical Docs Assistant (v0.2)")
st.caption("Restart-safe, versioned documentation assistant with document management.")

# --- SIDEBAR: Status, Doc Management & Upload ---
with st.sidebar:
    st.header("⚙️ Backend Status")
    try:
        health_resp = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            st.success(
                f"Online | Chunks: {health_data.get('total_chunks', 0)} | Docs: {health_data.get('total_documents', 0)}"
            )
        else:
            st.error("Backend Error")
    except requests.exceptions.ConnectionError:
        st.error("Backend Offline! Make sure uvicorn is running on port 8000.")

    st.divider()

    # --- Document List with Delete Buttons ---
    st.header("📑 Indexed Documents")
    try:
        docs_resp = requests.get(f"{BACKEND_URL}/documents", timeout=2)
        if docs_resp.status_code == 200:
            doc_list = docs_resp.json().get("documents", [])
            if not doc_list:
                st.info("No documents indexed yet.")
            else:
                for doc_name in doc_list:
                    col1, col2 = st.columns([0.8, 0.2])
                    col1.markdown(f"📄 `{doc_name}`")
                    if col2.button("🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                        del_resp = requests.delete(f"{BACKEND_URL}/documents/{doc_name}")
                        if del_resp.status_code == 200:
                            st.toast(f"Deleted {doc_name}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete: {del_resp.text}")
    except Exception:
        st.warning("Could not load document list.")

    st.divider()

    # --- Upload Documents (PDF, MD, TXT) ---
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload .md, .txt, or .pdf files",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True
    )

    if st.button("Index Uploaded Files", disabled=not uploaded_files):
        with st.spinner("Processing, embedding, and saving to disk..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
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
                st.error(f"Error during upload: {e}")

    st.divider()
    top_k = st.slider("Top-K Passages to Retrieve", min_value=1, max_value=8, value=4)


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['number']}] {s['doc']}** (Chunk ID: `{s['chunk_id']}` | Cosine Score: `{s['score']}`)"
                    )
                    st.code(s["text"], language="markdown")

# User Question Input
if user_query := st.chat_input("Ask a question about your docs..."):
    # 1. Show user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Query the backend
    with st.chat_message("assistant"):
        with st.spinner("Searching persistent index and generating answer..."):
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
                                st.markdown(
                                    f"**[{s['number']}] {s['doc']}** (Chunk ID: `{s['chunk_id']}` | Cosine Score: `{s['score']}`)"
                                )
                                st.code(s["text"], language="markdown")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Could not reach backend: {e}")