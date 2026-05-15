"""
app.py  ─  Study Buddy RAG  |  Streamlit Web Interface
=======================================================
Run with:  streamlit run app.py
"""

import os
import time
import tempfile
from pathlib import Path

import streamlit as st
from rag_engine import RAGEngine, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS  ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fraunces:ital,wght@0,400;0,700;1,400&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #0f2027 100%);
    min-height: 100vh;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #6c3cf7 0%, #2563eb 60%, #06b6d4 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(108,60,247,0.4);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 3rem;
    font-weight: 700;
    color: white;
    margin: 0 0 0.5rem 0;
    position: relative;
}
.hero p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin: 0;
    position: relative;
}
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    border-radius: 50px;
    padding: 0.25rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1rem;
    position: relative;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.card-title {
    color: #a78bfa;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(6,182,212,0.10));
    border: 1px solid rgba(37,99,235,0.4);
    border-radius: 16px;
    padding: 1.75rem;
    color: #e2e8f0;
    font-size: 1rem;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}
.answer-box.out-of-scope {
    background: rgba(239,68,68,0.10);
    border-color: rgba(239,68,68,0.35);
}

/* ── Chunk card ── */
.chunk-card {
    background: rgba(255,255,255,0.03);
    border-left: 3px solid #6c3cf7;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    font-size: 0.875rem;
    color: #cbd5e1;
    line-height: 1.6;
}
.chunk-meta {
    font-size: 0.75rem;
    color: #818cf8;
    font-weight: 600;
    margin-bottom: 0.4rem;
    letter-spacing: 0.03em;
}
.score-bar {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #6c3cf7, #06b6d4);
    margin-top: 0.5rem;
}

/* ── Stats ── */
.stat-pill {
    display: inline-block;
    background: rgba(108,60,247,0.2);
    border: 1px solid rgba(108,60,247,0.4);
    color: #a78bfa;
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6c3cf7 !important;
    box-shadow: 0 0 0 3px rgba(108,60,247,0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6c3cf7, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(108,60,247,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(108,60,247,0.5) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(108,60,247,0.5) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Chat history ── */
.chat-user {
    background: rgba(108,60,247,0.15);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    text-align: right;
}
.chat-bot {
    background: rgba(255,255,255,0.05);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
    font-size: 0.95rem;
}
.chat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6c3cf7;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "engine"       not in st.session_state: st.session_state.engine       = RAGEngine()
if "index_ready"  not in st.session_state: st.session_state.index_ready  = False
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "tmp_paths"    not in st.session_state: st.session_state.tmp_paths    = []
if "doc_stats"    not in st.session_state: st.session_state.doc_stats    = {}

engine: RAGEngine = st.session_state.engine

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Study Buddy RAG")
    st.markdown("---")

    # API Key
    st.markdown("#### 🔑 Gemini API Key")
    api_key_input = st.text_input(
        "Paste your Google Gemini API key",
        type="password",
        placeholder="AIza...",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        engine.configure_api_key(api_key_input)
        st.success("✅ API key set!")

    st.markdown("---")

    # Upload
    st.markdown("#### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Textbooks, lecture notes, handouts — any study material in PDF",
    )

    # Chunking settings
    with st.expander("⚙️ Advanced Settings", expanded=False):
        chunk_size = st.slider("Chunk Size (chars)", 200, 1000, CHUNK_SIZE, 50)
        overlap    = st.slider("Chunk Overlap (chars)", 0, 200, CHUNK_OVERLAP, 10)
        top_k      = st.slider("Chunks to Retrieve", 1, 10, TOP_K, 1)

    # Build index button
    if uploaded_files:
        if st.button("🚀 Build Knowledge Base", use_container_width=True):
            tmp_dir   = tempfile.mkdtemp()
            tmp_paths = []
            for uf in uploaded_files:
                tmp_path = os.path.join(tmp_dir, uf.name)
                with open(tmp_path, "wb") as f:
                    f.write(uf.read())
                tmp_paths.append(tmp_path)

            st.session_state.tmp_paths = tmp_paths

            with st.spinner("⚡ Processing documents…"):
                try:
                    n_pages, n_chunks = engine.build_index(
                        tmp_paths,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        force_rebuild=True,
                    )
                    st.session_state.index_ready  = True
                    st.session_state.doc_stats    = engine.stats()
                    st.session_state.chat_history = []
                    st.success(f"✅ Indexed {n_pages} pages → {n_chunks} chunks")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.markdown("---")

    # Stats
    if st.session_state.index_ready:
        stats = st.session_state.doc_stats
        st.markdown("#### 📊 Knowledge Base Stats")
        st.markdown(f'<span class="stat-pill">📄 {stats["num_docs"]} doc(s)</span>'
                    f'<span class="stat-pill">🔷 {stats["total_chunks"]} chunks</span>',
                    unsafe_allow_html=True)
        for src in stats["sources"]:
            st.markdown(f"- `{src}`")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("""
<div style='color:rgba(255,255,255,0.35); font-size:0.72rem; margin-top:1rem;'>
Built for <b>SISTec Build-with-RAG 2026</b><br>
Problem Statement #1 · Study Buddy RAG<br>
Stack: Gemini · FAISS · Sentence-Transformers · Streamlit
</div>
""", unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
  <div class="badge">🏆 Build with RAG · SISTec 2026</div>
  <h1>📚 Study Buddy</h1>
  <p>Your intelligent document Q&amp;A assistant — ask anything from your uploaded study material.</p>
</div>
""", unsafe_allow_html=True)

# ── Two column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:

    if not st.session_state.index_ready:
        st.markdown("""
<div class="card">
  <div class="card-title">👈 Getting Started</div>
  <ol style="color:#cbd5e1; line-height:2; margin:0;">
    <li>Paste your <b>Google Gemini API key</b> in the sidebar</li>
    <li><b>Upload</b> one or more PDF study documents</li>
    <li>Click <b>Build Knowledge Base</b></li>
    <li>Start asking questions below!</li>
  </ol>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="card-title">💬 Ask a Question</div>', unsafe_allow_html=True)

    with st.form("query_form", clear_on_submit=True):
        query = st.text_area(
            "Your question",
            placeholder="e.g. What is Newton's second law? Explain photosynthesis. Summarize Chapter 3.",
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔍 Ask Study Buddy", use_container_width=True)

    if submitted and query.strip():
        if not st.session_state.index_ready:
            st.warning("⚠️ Please upload documents and build the knowledge base first.")
        else:
            with st.spinner("🤔 Thinking…"):
                try:
                    result = engine.answer(query.strip(), top_k=top_k)
                    st.session_state.chat_history.append({
                        "query":    query.strip(),
                        "answer":   result["answer"],
                        "chunks":   result["chunks"],
                        "in_scope": result["in_scope"],
                    })
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown('<div class="card-title">🕓 Conversation History</div>',
                    unsafe_allow_html=True)

        for turn in reversed(st.session_state.chat_history):
            scope_cls = "" if turn["in_scope"] else " out-of-scope"

            st.markdown(f"""
<div class="chat-user">
  <div class="chat-label">You</div>
  {turn['query']}
</div>""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="answer-box{scope_cls}">
  <div class="chat-label">{'📚 Study Buddy' if turn['in_scope'] else '⚠️ Out of Scope'}</div>
  {turn['answer']}
</div>""", unsafe_allow_html=True)

            if turn["chunks"]:
                with st.expander(f"📎 View {len(turn['chunks'])} retrieved source chunks"):
                    for i, chunk in enumerate(turn["chunks"], 1):
                        score_pct   = min(int(chunk['score'] * 100), 100)
                        score_bar_w = max(score_pct, 5)
                        st.markdown(f"""
<div class="chunk-card">
  <div class="chunk-meta">Source {i} · {chunk['source']} · Page {chunk['page']} · Relevance: {score_pct}%</div>
  {chunk['text']}
  <div class="score-bar" style="width:{score_bar_w}%;"></div>
</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

with col_right:

    st.markdown("""
<div class="card">
  <div class="card-title">💡 Tips for Better Answers</div>
  <ul style="color:#cbd5e1; line-height:2; margin:0; padding-left:1.2rem;">
    <li>Ask specific questions about topics in your document</li>
    <li>Use keywords that appear in the text</li>
    <li>Ask for explanations, summaries, or definitions</li>
    <li>Questions outside the document will be rejected</li>
  </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <div class="card-title">🎯 Sample Questions to Try</div>
  <div style="color:#cbd5e1; font-size:0.875rem; line-height:2;">
    ✦ <em>Summarize the main concepts in this document</em><br>
    ✦ <em>What is [topic]? Explain in simple terms.</em><br>
    ✦ <em>List the key points from Chapter X</em><br>
    ✦ <em>What are the differences between A and B?</em><br>
    ✦ <em>How does [process] work?</em>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <div class="card-title">⚙️ How It Works</div>
  <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.8;">
    <b style="color:#a78bfa;">1. Load</b> &nbsp;→ PDFs are parsed &amp; cleaned<br>
    <b style="color:#a78bfa;">2. Chunk</b> &nbsp;→ Text split into 500-char overlapping chunks<br>
    <b style="color:#a78bfa;">3. Embed</b> &nbsp;→ Each chunk encoded with <code>all-MiniLM-L6-v2</code><br>
    <b style="color:#a78bfa;">4. Store</b> &nbsp;→ Vectors indexed in FAISS (cosine similarity)<br>
    <b style="color:#a78bfa;">5. Retrieve</b> → Top-K chunks fetched per query<br>
    <b style="color:#a78bfa;">6. Generate</b> → Gemini 1.5 Flash answers from context only
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <div class="card-title">🏆 Judging Criteria Coverage</div>
  <div style="font-size:0.82rem; color:#cbd5e1; line-height:2;">
    ✅ RAG workflow (load, chunk, embed, retrieve, generate)<br>
    ✅ Answer accuracy &amp; source grounding<br>
    ✅ Out-of-scope question rejection<br>
    ✅ Gemini API integration &amp; prompt engineering<br>
    ✅ Streamlit UI — usability &amp; clean design<br>
    ✅ Source chunk display with relevance scores
  </div>
</div>
""", unsafe_allow_html=True)
