import os
import time
import requests
import streamlit as st
from typing import Dict, Any, List

# ==============================================================================
# Streamlit Application Configuration
# ==============================================================================
st.set_page_config(
    page_title="CampusRAG | German University Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")

# ==============================================================================
# Custom CSS for Modern, Premium Dark-Mode Glassmorphism Design
# ==============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient & Card */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 6px;
        line-height: 1.5;
    }
    .hero-badges {
        margin-top: 14px;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .badge-rag { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.35); }
    .badge-gdpr { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-local { background: rgba(168, 85, 247, 0.18); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }
    .badge-chroma { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }

    /* Source Citation Cards */
    .citation-card {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
        font-weight: 600;
        color: #e2e8f0;
    }
    .citation-meta {
        color: #38bdf8;
        font-size: 0.8rem;
        font-family: monospace;
    }
    .citation-score {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
    }
    .citation-body {
        color: #cbd5e1;
        font-style: italic;
        line-height: 1.4;
    }

    /* Guardrail Status Meters */
    .status-grounded {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-partial {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-refusal {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Helper Functions to Communicate with FastAPI Backend
# ==============================================================================
def check_backend_health() -> Dict[str, Any]:
    try:
        res = requests.get(f"{BACKEND_URL}/health", timeout=3.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"status": "unreachable", "documents_indexed": 0, "total_chunks": 0}

def fetch_documents() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{BACKEND_URL}/documents", timeout=3.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []

def query_rag_backend(query_text: str, top_k: int, threshold: float, doc_filter: str = None) -> Dict[str, Any]:
    payload = {
        "query": query_text,
        "top_k": top_k,
        "similarity_threshold": threshold,
        "filter_doc": doc_filter if doc_filter != "All Documents" else None,
    }
    try:
        res = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=60.0)
        if res.status_code == 200:
            return res.json()
        return {"answer": f"Error from backend (HTTP {res.status_code}): {res.text}", "sources": [], "confidence_score": 0.0, "guardrail_status": "ERROR", "latency_ms": 0, "model_used": "N/A"}
    except requests.exceptions.ConnectionError:
        return {
            "answer": f"⚠️ Could not connect to FastAPI backend at {BACKEND_URL}. Ensure the backend is running (`uvicorn app.main:app --port 8000`).",
            "sources": [],
            "confidence_score": 0.0,
            "guardrail_status": "OFFLINE",
            "latency_ms": 0,
            "model_used": "Offline",
        }
    except Exception as e:
        return {"answer": f"Request exception: {str(e)}", "sources": [], "confidence_score": 0.0, "guardrail_status": "ERROR", "latency_ms": 0, "model_used": "N/A"}

def upload_file_to_backend(uploaded_file) -> Dict[str, Any]:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
        res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60.0)
        return res.json() if res.status_code == 200 else {"status": "FAILED", "message": res.text}
    except Exception as e:
        return {"status": "FAILED", "message": str(e)}

def load_sample_docs_to_backend() -> Dict[str, Any]:
    try:
        res = requests.post(f"{BACKEND_URL}/ingest-sample-docs", timeout=60.0)
        return res.json() if res.status_code == 200 else []
    except Exception as e:
        st.error(f"Failed to ingest sample docs: {e}")
        return []

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Servus! I am the **Campus Knowledge Assistant**. Ask me anything about examination rules, ECTS requirements, library policies, or semester ticket guidelines. Every answer is strictly grounded with verifiable document citations.",
            "sources": [],
            "confidence": 1.0,
            "status": "PASS_GROUNDED",
            "latency": 0.0,
            "model": "System Init",
        }
    ]

# ==============================================================================
# Sidebar: System Telemetry, Ingestion & Retrieval Fine-Tuning
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/graduation-cap.png", width=64)
    st.title("Campus Knowledge Base")
    st.caption("Production RAG Architecture | German Enterprise Standards")

    # Live Backend Status
    health = check_backend_health()
    if health["status"] == "healthy":
        st.success(f"🟢 **Backend Connected** | API v{health.get('version', '1.0')}")
        st.info(
            f"📊 **Vectors Indexed:** {health.get('total_chunks', 0)} chunks across "
            f"{health.get('documents_indexed', 0)} documents\n\n"
            f"🧠 **Embedding Model:** `{health.get('embedding_model', 'all-MiniLM-L6-v2')}`\n\n"
            f"🤖 **Active LLM:** `{health.get('active_llm_model', 'Ollama / Local')}`"
        )
    else:
        st.error("🔴 **Backend Unreachable**\nRun `uvicorn app.main:app` in `backend/`")

    st.divider()

    # RAG Retrieval Configuration
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider("Top-K Chunks (Precision Pool)", min_value=1, max_value=8, value=4, help="Number of semantic chunks fetched for LLM context.")
    threshold = st.slider("Cosine Similarity Threshold", min_value=0.1, max_value=0.9, value=0.35, step=0.05, help="Cutoff threshold: chunks below this score are discarded as irrelevant.")

    # Document Filtering
    doc_list = fetch_documents()
    doc_options = ["All Documents"] + [d["document_name"] for d in doc_list]
    selected_doc_filter = st.selectbox("Filter by Specific Document", options=doc_options)

    st.divider()

    # Document Ingestion Panel
    st.subheader("📥 Ingestion Pipeline")
    uploaded_file = st.file_uploader("Upload Campus Doc (PDF, MD, TXT)", type=["pdf", "md", "txt"])
    if uploaded_file is not None:
        if st.button("Index Uploaded Document", use_container_width=True):
            with st.spinner("Processing & chunking document..."):
                res = upload_file_to_backend(uploaded_file)
                if res.get("status") == "SUCCESS":
                    st.success(f"✅ {res.get('message')}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Ingestion failed: {res.get('message')}")

    if st.button("⚡ Index Sample University Docs", use_container_width=True, help="Load TUM FPSO exam regulations, library rules, AI curriculum, and semester ticket guides."):
        with st.spinner("Indexing bundled campus documents into ChromaDB..."):
            results = load_sample_docs_to_backend()
            st.success(f"Indexed {len(results)} sample university documents!")
            time.sleep(1)
            st.rerun()

    # Document Inventory
    if doc_list:
        with st.expander(f"📚 Indexed Corpus ({len(doc_list)} files)"):
            for d in doc_list:
                st.markdown(f"**{d['document_name']}**")
                st.caption(f"{d['chunks_count']} chunks • {d['pages_count']} pages")
                st.markdown("---")

# ==============================================================================
# Main View: Header & Architectural Highlights
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1 class="hero-title">🎓 Campus Knowledge Assistant</h1>
        <div class="hero-subtitle">
            Enterprise-grade Retrieval-Augmented Generation (RAG) system with strict source citations,
            cosine similarity thresholding, and GDPR-compliant local inference.
        </div>
        <div class="hero-badges">
            <span class="badge badge-rag">⚡ Dense Vector RAG</span>
            <span class="badge badge-gdpr">🔒 GDPR-Compliant Local Inference</span>
            <span class="badge badge-local">🧠 sentence-transformers (all-MiniLM-L6-v2)</span>
            <span class="badge badge-chroma">💾 Persistent ChromaDB HNSW</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Query Chips
st.markdown("**Sample Campus Queries (Click to test):**")
col1, col2, col3, col4 = st.columns(4)
sample_query = None

with col1:
    if st.button("🇩🇪 Ärztliches Attest Frist", use_container_width=True):
        sample_query = "Bis wann muss ein ärztliches Attest nach einer verpassten Prüfung eingereicht werden?"
with col2:
    if st.button("🇩🇪 ECTS Masterarbeit", use_container_width=True):
        sample_query = "Wie viele ECTS-Punkte umfasst die Masterarbeit und wie lange ist die Bearbeitungszeit?"
with col3:
    if st.button("🇬🇧 Library Food Policy", use_container_width=True):
        sample_query = "Can students bring food or coffee into the central library study halls?"
with col4:
    if st.button("🛡️ Out-of-Domain Refusal", use_container_width=True):
        sample_query = "What is the best recipe for baking chocolate chip cookies?"

# ==============================================================================
# Chat History Stream
# ==============================================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display guardrail metrics and verified citations if available
        if msg.get("sources") or msg.get("status"):
            status = msg.get("status", "PASS_GROUNDED")
            conf = msg.get("confidence", 0.0)
            latency = msg.get("latency", 0.0)
            model = msg.get("model", "N/A")

            # Status pill
            if status == "PASS_GROUNDED":
                status_html = f'<span class="status-grounded">🛡️ Verified Grounded ({int(conf * 100)}%)</span>'
            elif status == "PARTIAL_RELEVANCE":
                status_html = f'<span class="status-partial">⚠️ Partial Grounding ({int(conf * 100)}%)</span>'
            else:
                status_html = f'<span class="status-refusal">🚫 Refusal: Missing from Corpus (0%)</span>'

            st.markdown(
                f"""
                <div style="margin-top: 8px; margin-bottom: 8px; display: flex; gap: 12px; align-items: center; font-size: 0.8rem; color: #94a3b8;">
                    {status_html}
                    <span>⏱️ {latency} ms</span>
                    <span>🤖 {model}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display collapsible verifiable source citations
            if msg.get("sources"):
                with st.expander(f"📑 View {len(msg['sources'])} Verifiable Document Citations"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-meta">📄 {s['document_name']} • Page {s['page_number']}</span>
                                    <span class="citation-score">Cosine Sim: {s['similarity_score']:.2f}</span>
                                </div>
                                <div class="citation-body">"{s['excerpt']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ==============================================================================
# Chat Input & RAG Query Execution
# ==============================================================================
user_prompt = st.chat_input("Ask a question about German university guidelines, exam rules, or library services...")

# Handle sample button clicks
if sample_query:
    user_prompt = sample_query

if user_prompt:
    # 1. Append user question to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Query backend RAG engine
    with st.chat_message("assistant"):
        with st.spinner("Retrieving verified campus documents & generating grounded response..."):
            res = query_rag_backend(
                query_text=user_prompt,
                top_k=top_k,
                threshold=threshold,
                doc_filter=selected_doc_filter,
            )

        answer_text = res.get("answer", "")
        sources = res.get("sources", [])
        confidence = res.get("confidence_score", 0.0)
        status = res.get("guardrail_status", "PASS_GROUNDED")
        latency = res.get("latency_ms", 0.0)
        model = res.get("model_used", "Local RAG")

        st.markdown(answer_text)

        # Status indicator
        if status == "PASS_GROUNDED":
            status_html = f'<span class="status-grounded">🛡️ Verified Grounded ({int(confidence * 100)}%)</span>'
        elif status == "PARTIAL_RELEVANCE":
            status_html = f'<span class="status-partial">⚠️ Partial Grounding ({int(confidence * 100)}%)</span>'
        else:
            status_html = f'<span class="status-refusal">🚫 Refusal: Missing from Corpus (0%)</span>'

        st.markdown(
            f"""
            <div style="margin-top: 8px; margin-bottom: 8px; display: flex; gap: 12px; align-items: center; font-size: 0.8rem; color: #94a3b8;">
                {status_html}
                <span>⏱️ {latency} ms</span>
                <span>🤖 {model}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if sources:
            with st.expander(f"📑 View {len(sources)} Verifiable Document Citations"):
                for s in sources:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-meta">📄 {s['document_name']} • Page {s['page_number']}</span>
                                <span class="citation-score">Cosine Sim: {s['similarity_score']:.2f}</span>
                            </div>
                            <div class="citation-body">"{s['excerpt']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # 3. Append assistant response to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "confidence": confidence,
                "status": status,
                "latency": latency,
                "model": model,
            }
        )
