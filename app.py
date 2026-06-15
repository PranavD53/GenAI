import streamlit as st
import faiss
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS — Deep Navy / Frosted Glass Design
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #070E1C !important;
    font-family: 'Inter', sans-serif;
    color: #E2EAF4;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1628 0%, #0D1E38 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 2rem 1.5rem !important;
}

/* ── Sidebar Logo Area ── */
.sb-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sb-logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #1D4ED8, #38BDF8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 0 18px rgba(56,189,248,0.35);
}
.sb-logo-text { line-height: 1.2; }
.sb-logo-title { font-size: 0.95rem; font-weight: 700; color: #F0F4FF; letter-spacing: 0.01em; }
.sb-logo-sub { font-size: 0.7rem; color: #7896B8; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Sidebar Section Labels ── */
.sb-section-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #4A6580;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem;
    margin-top: 1.75rem;
}
.sb-section-label:first-of-type { margin-top: 0; }

/* ── Category Pills ── */
.cat-pill {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.9rem;
    border-radius: 8px;
    margin-bottom: 0.35rem;
    cursor: pointer;
    transition: background 0.18s ease;
    font-size: 0.82rem; font-weight: 500;
    color: #8AAAC8;
    border: 1px solid transparent;
}
.cat-pill:hover {
    background: rgba(37, 99, 235, 0.12);
    color: #38BDF8;
    border-color: rgba(56, 189, 248, 0.18);
}
.cat-pill .icon { font-size: 1rem; }

/* ── Stats Box ── */
.sb-stat {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-top: 1.5rem;
}
.sb-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0;
    font-size: 0.78rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sb-stat-row:last-child { border-bottom: none; padding-bottom: 0; }
.sb-stat-label { color: #5C7A95; }
.sb-stat-value { color: #38BDF8; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* ── Main Content Wrapper ── */
.main-wrapper {
    padding: 3rem 4rem 4rem;
    max-width: 960px;
    margin: 0 auto;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    margin-bottom: 3.5rem;
    padding-top: 1rem;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: #60A5FA;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.25rem;
}
.hero-badge .dot {
    width: 6px; height: 6px;
    background: #38BDF8;
    border-radius: 50%;
    box-shadow: 0 0 8px #38BDF8;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    line-height: 1.15;
    color: #F0F4FF;
    letter-spacing: -0.02em;
    margin-bottom: 0.85rem;
}
.hero h1 span {
    background: linear-gradient(90deg, #38BDF8, #6366F1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem;
    color: #6B90B0;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Search Container ── */
.search-wrap {
    position: relative;
    margin-bottom: 2rem;
}
.search-icon {
    position: absolute;
    left: 1.1rem; top: 50%;
    transform: translateY(-50%);
    color: #3B6EA5;
    font-size: 1.1rem;
    z-index: 1;
    pointer-events: none;
}

/* Override Streamlit text input */
[data-testid="stTextInput"] {
    margin-bottom: 0 !important;
}
[data-testid="stTextInput"] > div > div {
    background: rgba(255,255,255,0.035) !important;
    border: 1.5px solid rgba(56,189,248,0.2) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stTextInput"] > div > div:focus-within {
    border-color: rgba(56,189,248,0.55) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.09), 0 0 30px rgba(56,189,248,0.06) !important;
}
[data-testid="stTextInput"] input {
    color: #E2EAF4 !important;
    font-size: 0.97rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 1rem 1.1rem 1rem 3rem !important;
    background: transparent !important;
    caret-color: #38BDF8 !important;
}
[data-testid="stTextInput"] input::placeholder { color: #3D5E7A !important; }
[data-testid="stTextInput"] label { display: none !important; }

/* ── Quick-topic chips ── */
.chips-row {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    margin-bottom: 2.5rem;
    justify-content: center;
}
.chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 0.38rem 0.85rem;
    font-size: 0.78rem;
    color: #7896B8;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.chip:hover {
    border-color: rgba(56,189,248,0.3);
    color: #38BDF8;
    background: rgba(56,189,248,0.06);
}

/* ── Spinner override ── */
[data-testid="stSpinner"] > div { color: #38BDF8 !important; }
.stSpinner { border-color: #38BDF8 transparent transparent transparent !important; }

/* ── Answer Card ── */
.answer-card {
    background: linear-gradient(135deg, rgba(15,30,58,0.9) 0%, rgba(10,20,44,0.95) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
}
.answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #1D4ED8, #38BDF8, #6366F1);
}
.answer-label {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    color: #38BDF8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.answer-label .ai-dot {
    width: 8px; height: 8px;
    background: #38BDF8;
    border-radius: 50%;
    box-shadow: 0 0 10px #38BDF8;
}
.answer-text {
    font-size: 1.02rem;
    line-height: 1.75;
    color: #D4E4F5;
    font-weight: 400;
}

/* ── Context Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #5C7A95 !important;
    padding: 0.85rem 1.1rem !important;
}
[data-testid="stExpander"] summary:hover { color: #90B4CC !important; }
.context-block {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    color: #8AAAC8;
}
.context-block:last-child { margin-bottom: 0; }

/* ── Metadata Row ── */
.meta-row {
    display: flex; align-items: center; gap: 1.5rem;
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    flex-wrap: wrap;
}
.meta-item {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.73rem; color: #3D5E7A;
}
.meta-item .val { color: #60A5FA; font-weight: 500; }

/* ── Divider ── */
.styled-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.15), transparent);
    margin: 2rem 0;
}

/* ── Confidence Bar ── */
.conf-row {
    display: flex; align-items: center; gap: 0.75rem;
    margin-top: 1rem;
}
.conf-label { font-size: 0.73rem; color: #4A6580; min-width: 80px; }
.conf-bar-bg {
    flex: 1; height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #1D4ED8, #38BDF8);
    border-radius: 100px;
    transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
}
.conf-pct { font-size: 0.73rem; color: #38BDF8; font-weight: 600; font-family: 'JetBrains Mono', monospace; min-width: 36px; text-align: right; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 16px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.45) !important;
}

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #2E4A63;
}
.empty-state .big-icon {
    font-size: 3.5rem;
    margin-bottom: 1.25rem;
    opacity: 0.4;
}
.empty-state p { font-size: 0.88rem; line-height: 1.6; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.2); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Models & Assets Once
# --------------------------------------------------
@st.cache_resource
def load_assets():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("employee_policy_faiss.index")
    with open("chunk_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)
    return embed_model, index, mapping, tokenizer, model

embed_model, index, mapping, tokenizer, llm = load_assets()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🏢</div>
        <div class="sb-logo-text">
            <div class="sb-logo-title">PolicyAI</div>
            <div class="sb-logo-sub">HR Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-label">Browse by category</div>', unsafe_allow_html=True)

    categories = [
        ("🏖️", "Leave Policy"),
        ("🏠", "Work From Home"),
        ("🏥", "Insurance & Health"),
        ("🎁", "Employee Benefits"),
        ("📋", "Company Rules"),
        ("💰", "Compensation"),
        ("🎓", "Learning & Development"),
        ("🔒", "Data & Privacy"),
    ]
    for icon, label in categories:
        st.markdown(f"""
        <div class="cat-pill">
            <span class="icon">{icon}</span>{label}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-stat">
        <div class="sb-stat-row">
            <span class="sb-stat-label">Documents indexed</span>
            <span class="sb-stat-value">12</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-label">Policy chunks</span>
            <span class="sb-stat-value">1,847</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-label">Model</span>
            <span class="sb-stat-value">flan-t5</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-label">Embeddings</span>
            <span class="sb-stat-value">MiniLM</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Main Content
# --------------------------------------------------
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="dot"></div>
        AI-Powered · Policy RAG System
    </div>
    <h1>Ask anything about<br><span>company policy</span></h1>
    <p class="hero-sub">
        Instant, grounded answers from your official HR documents —
        no guessing, no searching through PDFs.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick topic chips
st.markdown("""
<div class="chips-row">
    <div class="chip">🏖️ Casual leave count</div>
    <div class="chip">🏠 WFH eligibility</div>
    <div class="chip">🏥 Maternity policy</div>
    <div class="chip">💊 Medical insurance</div>
    <div class="chip">💰 Appraisal cycle</div>
    <div class="chip">📋 Notice period</div>
    <div class="chip">🎓 Training budget</div>
</div>
""", unsafe_allow_html=True)

# Search input with icon
st.markdown('<div class="search-icon">🔍</div>', unsafe_allow_html=True)
query = st.text_input(
    "query",
    placeholder="e.g. How many sick leaves am I entitled to per year?",
    key="main_query",
    label_visibility="collapsed"
)

# --------------------------------------------------
# Processing & Answer
# --------------------------------------------------
if query.strip():
    t0 = time.time()

    with st.spinner("Searching policy documents…"):
        q_emb = embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, k=4)

        context_chunks = []
        scores = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                context_chunks.append(mapping[idx])
                scores.append(float(dist))

        context = "\n\n".join(context_chunks)

    with st.spinner("Generating answer with AI…"):
        prompt = f"""You are a precise HR Policy Assistant.
Answer ONLY using the provided context. Be concise and direct.
If the answer is not present in the context, say: "I could not find that information in the policy documents."

Context:
{context}

Question: {query}
Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = llm.generate(**inputs, max_new_tokens=200, temperature=0.3, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elapsed = time.time() - t0
    avg_score = np.mean(scores) if scores else 0
    confidence_pct = min(int(avg_score * 100), 99)

    # Answer card
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">
            <div class="ai-dot"></div>
            AI Response
        </div>
        <div class="answer-text">{answer}</div>
        <div class="conf-row">
            <span class="conf-label">Confidence</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence_pct}%"></div>
            </div>
            <span class="conf-pct">{confidence_pct}%</span>
        </div>
        <div class="meta-row">
            <div class="meta-item">⏱ Response time: <span class="val">{elapsed:.1f}s</span></div>
            <div class="meta-item">📑 Chunks retrieved: <span class="val">{len(context_chunks)}</span></div>
            <div class="meta-item">🔍 Query: <span class="val">{query[:40]}{'…' if len(query)>40 else ''}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Context expander
    with st.expander("📄 View retrieved policy excerpts"):
        for i, (chunk, score) in enumerate(zip(context_chunks, scores)):
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                <span style="font-size:0.68rem;color:#3D5E7A;text-transform:uppercase;letter-spacing:0.08em;font-weight:600;">
                    Source {i+1}
                </span>
                <span style="font-size:0.68rem;color:#1D4ED8;font-family:'JetBrains Mono',monospace;">
                    score {score:.3f}
                </span>
            </div>
            <div class="context-block">{chunk}</div>
            """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">🔍</div>
        <p>Type a question above to get an instant answer<br>grounded in your company's official policy documents.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-wrapper
