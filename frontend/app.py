import os
import time
import requests
import streamlit as st

# Fallback to localhost:8000 for local runs, or http://backend:8000 if inside Docker
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if "backend" in BACKEND_URL and not os.path.exists("/.dockerenv"):
    BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="DocPilot v2.0", page_icon="📚", layout="wide")

st.title("📚 DocPilot: Technical Docs Assistant (v2.0)")
st.caption("Autonomous, restart-safe, multi-turn documentation assistant with verified citations.")

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
    # --- Metadata Filter Dropdown ---
    st.header("🔍 Search Filters")
    try:
        docs_resp = requests.get(f"{BACKEND_URL}/documents", timeout=2)
        if docs_resp.status_code == 200:
            all_docs = docs_resp.json().get("documents", [])
            doc_options = ["All Documents"] + all_docs
            selected_doc = st.selectbox("Restrict Search to Document:", doc_options)
            active_doc_filter = None if selected_doc == "All Documents" else selected_doc
        else:
            active_doc_filter = None
    except Exception:
        active_doc_filter = None

    top_k = st.slider("Top-K Passages to Retrieve", min_value=1, max_value=8, value=4)


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            st.markdown("##### 📌 Verified Quotes")
            for c in msg["citations"]:
                st.info(f"**Passage [{c['passage_number']}]:** \"{c['exact_quote']}\"")
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['number']}] {s['doc']}** (Relevance Score: `{s.get('score', 0)}`)"
                    )
                    st.code(s["text"], language="markdown")

# User Question Input
if user_query := st.chat_input("Ask a question about your docs..."):
    # 1. Show user question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Query backend agent
    with st.chat_message("assistant"):
        resp_data = None

        # Animated status box (closes before displaying answer)
        with st.status("🤖 Agent thinking & retrieving...", expanded=True) as status:
            status.write("🔍 Resolving follow-ups & searching Qdrant...")
            time.sleep(0.2)
            status.write("📊 Grading document relevance...")
            time.sleep(0.2)
            status.write("🛡️ Verifying citations & checking groundedness (Self-RAG)...")

            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={
                        "question": user_query,
                        "top_k": top_k,
                        "doc_filter": active_doc_filter,
                        "chat_history": st.session_state.messages[-4:]
                    },
                    timeout=45
                )
                if resp.status_code == 200:
                    resp_data = resp.json()
                    status.update(label="✅ Answer verified & grounded!", state="complete", expanded=False)
                else:
                    status.update(label=f"❌ Error ({resp.status_code})", state="error", expanded=False)
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                status.update(label="❌ Connection Failed", state="error", expanded=False)
                st.error(f"Could not reach backend: {e}")

        # 3. Render Answer, Quotes, and Sources OUTSIDE the status box!
        if resp_data:
            answer = resp_data.get("answer", "")
            sources = resp_data.get("sources", [])
            citations = resp_data.get("citations", [])

            st.markdown(answer)

            # Verified Quotes Pills
            if citations:
                st.markdown("##### 📌 Verified Quotes")
                for c in citations:
                    st.info(f"**Passage [{c['passage_number']}]:** \"{c['exact_quote']}\"")

            # Retrieved Sources Accordion
            if sources:
                with st.expander("🔍 View Retrieved Sources"):
                    for s in sources:
                        st.markdown(f"**[{s['number']}] {s['doc']}** (Relevance Score: `{s.get('score', 0)}`)")
                        st.code(s["text"], language="markdown")

            # Save to conversation history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "citations": citations
            })