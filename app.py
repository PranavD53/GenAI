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
    page_title="Policy Assistant | Enterprise HR Platform",
    page_icon="static/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS Styling
# --------------------------------------------------
def load_custom_css():
    st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* CSS Variables */
        :root {
            --primary-50: #eff6ff;
            --primary-100: #dbeafe;
            --primary-200: #bfdbfe;
            --primary-300: #93c5fd;
            --primary-400: #60a5fa;
            --primary-500: #3b82f6;
            --primary-600: #2563eb;
            --primary-700: #1d4ed8;
            --primary-800: #1e40af;
            --primary-900: #1e3a8a;

            --accent-50: #f0fdfa;
            --accent-100: #ccfbf1;
            --accent-400: #2dd4bf;
            --accent-500: #14b8a6;
            --accent-600: #0d9488;

            --success-50: #f0fdf4;
            --success-500: #22c55e;
            --success-600: #16a34a;

            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
        }

        /* Global Styles */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        body {
            background: linear-gradient(135deg, var(--primary-50) 0%, #ffffff 50%, var(--accent-50) 100%) !important;
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Main Container */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1200px !important;
        }

        /* Header Styling */
        .app-header {
            background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px -10px rgba(37, 99, 235, 0.4);
            position: relative;
            overflow: hidden;
        }

        .app-header::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: translate(30%, -30%);
        }

        .app-header h1 {
            color: white !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }

        .app-header p {
            color: rgba(255,255,255,0.9) !important;
            font-size: 1rem !important;
            margin: 0 !important;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(255,255,255,0.2);
            padding: 0.5rem 1rem;
            border-radius: 999px;
            font-size: 0.875rem;
            color: white;
            margin-top: 1rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success-500);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Quick Action Cards */
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }

        .quick-action-card {
            background: white;
            border: 1px solid var(--gray-200);
            border-radius: 12px;
            padding: 1.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .quick-action-card:hover {
            border-color: var(--primary-300);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
            transform: translateY(-2px);
        }

        .quick-action-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--primary-100), var(--primary-200));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 0.75rem;
            font-size: 1.25rem;
        }

        .quick-action-title {
            font-weight: 600;
            color: var(--gray-800);
            margin-bottom: 0.25rem;
        }

        .quick-action-desc {
            font-size: 0.8125rem;
            color: var(--gray-500);
        }

        /* Input Area */
        .input-container {
            background: white;
            border: 2px solid var(--gray-200);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }

        .input-container:focus-within {
            border-color: var(--primary-400);
            box-shadow: 0 0 0 3px var(--primary-100), 0 4px 12px rgba(59, 130, 246, 0.1);
        }

        .input-label {
            font-weight: 600;
            color: var(--gray-700);
            margin-bottom: 0.75rem;
            display: block;
        }

        /* Text Input Styling */
        div[data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }

        div[data-testid="stTextInput"] > div > div > input {
            border: none !important;
            background: transparent !important;
            font-size: 1rem !important;
            padding: 0.75rem 0 !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] > div > div > input:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] > div > label {
            display: none !important;
        }

        /* Submit Button */
        .submit-button {
            background: linear-gradient(135deg, var(--primary-500), var(--primary-600)) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.875rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3) !important;
            width: 100% !important;
        }

        .submit-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }

        /* Response Container */
        .response-container {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid var(--gray-200);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            animation: slideUp 0.3s ease;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .response-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--gray-100);
        }

        .response-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--accent-400), var(--accent-500));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.25rem;
        }

        .response-title {
            font-weight: 600;
            color: var(--gray-800);
            font-size: 1.125rem;
        }

        .response-text {
            color: var(--gray-700);
            font-size: 1rem;
            line-height: 1.7;
            white-space: pre-wrap;
        }

        /* Success Box Override */
        .stSuccess {
            background: linear-gradient(135deg, var(--success-50), #ffffff) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            box-shadow: none !important;
        }

        .stSuccess > div {
            color: var(--gray-800) !important;
            font-size: 1rem !important;
            line-height: 1.7 !important;
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background: var(--gray-50) !important;
            border: 1px solid var(--gray-200) !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            font-weight: 600 !important;
            color: var(--gray-700) !important;
            margin-top: 1rem !important;
        }

        .streamlit-expanderContent {
            border: 1px solid var(--gray-200) !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 1.25rem !important;
            background: var(--gray-50) !important;
        }

        /* Loading Animation */
        .loading-container {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
        }

        .loading-dots {
            display: flex;
            gap: 4px;
        }

        .loading-dot {
            width: 8px;
            height: 8px;
            background: var(--primary-400);
            border-radius: 50%;
            animation: bounce 1s infinite ease-in-out;
        }

        .loading-dot:nth-child(1) { animation-delay: 0s; }
        .loading-dot:nth-child(2) { animation-delay: 0.15s; }
        .loading-dot:nth-child(3) { animation-delay: 0.3s; }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: white !important;
            border-right: 1px solid var(--gray-200) !important;
        }

        section[data-testid="stSidebar"] > div {
            padding: 1.5rem 1rem !important;
        }

        .sidebar-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--gray-500);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            padding: 0 0.5rem;
        }

        .sidebar-category {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.875rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }

        .sidebar-category:hover {
            background: var(--primary-50);
            border-color: var(--primary-200);
        }

        .sidebar-cat-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1rem;
        }

        .sidebar-cat-text {
            display: flex;
            flex-direction: column;
        }

        .sidebar-cat-name {
            font-weight: 600;
            color: var(--gray-800);
            font-size: 0.875rem;
        }

        .sidebar-cat-count {
            font-size: 0.75rem;
            color: var(--gray-500);
        }

        /* Chat History */
        .chat-message {
            background: white;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border: 1px solid var(--gray-200);
            display: flex;
            gap: 1rem;
        }

        .chat-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .chat-avatar.user {
            background: linear-gradient(135deg, var(--primary-400), var(--primary-500));
            color: white;
        }

        .chat-avatar.assistant {
            background: linear-gradient(135deg, var(--accent-400), var(--accent-500));
            color: white;
        }

        .chat-content {
            flex: 1;
        }

        .chat-role {
            font-weight: 600;
            font-size: 0.8125rem;
            color: var(--gray-700);
            margin-bottom: 0.25rem;
        }

        .chat-text {
            color: var(--gray-600);
            font-size: 0.875rem;
            line-height: 1.5;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem !important;
            }

            .app-header {
                padding: 1.5rem;
                border-radius: 12px;
            }

            .app-header h1 {
                font-size: 1.5rem !important;
            }

            .quick-actions {
                grid-template-columns: 1fr;
            }
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--gray-300);
            border-radius: 999px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--gray-400);
        }

        /* Tooltip Button */
        .tooltip-btn {
            background: transparent;
            border: 1px solid var(--gray-300);
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            font-size: 0.8125rem;
            color: var(--gray-600);
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.375rem;
        }

        .tooltip-btn:hover {
            background: var(--primary-50);
            border-color: var(--primary-300);
            color: var(--primary-700);
        }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# --------------------------------------------------
# Load Models & Assets Once
# --------------------------------------------------
@st.cache_resource
def load_assets():
    # Embedding Model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    # FAISS Index
    try:
        index = faiss.read_index("employee_policy_faiss.index")
        with open("chunk_mapping.pkl", "rb") as f:
            mapping = pickle.load(f)
    except FileNotFoundError:
        # Demo mode if index not found
        index = None
        mapping = {}
    # Lightweight LLM
    model_id = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    return embed_model, index, mapping, tokenizer, model

embed_model, index, mapping, tokenizer, llm = load_assets()

# --------------------------------------------------
# Demo Responses
# --------------------------------------------------
DEMO_RESPONSES = {
    "leave": {
        "answer": "Employees are entitled to 12 casual leaves per calendar year. These leaves are credited at the beginning of each year on a pro-rata basis. Unused casual leaves cannot be carried forward to the next year and will lapse at year-end.",
        "context": "Policy Manual Section 4.2 - Leave Entitlements\n\nAll permanent employees are eligible for:\n1. Casual Leave: 12 days per year\n2. Sick Leave: 6 days per year\n3. Earned Leave: 15 days per year\n4. Maternity Leave: 26 weeks\n5. Paternity Leave: 10 days"
    },
    "wfh": {
        "answer": "The Work From Home policy allows employees to work remotely up to 3 days per week with manager approval. Employees must maintain regular working hours and be available during core hours (10 AM - 4 PM).",
        "context": "Policy Manual Section 5.1 - Remote Work Guidelines\n\n- Maximum 3 days per week WFH\n- Prior manager approval required\n- Core hours: 10 AM - 4 PM\n- Request must be submitted 24 hours in advance"
    },
    "insurance": {
        "answer": "The company provides comprehensive health insurance for all permanent employees. Coverage includes hospitalization, outpatient care, prescription medications, and preventive health check-ups. Sum insured is Rs. 5 lakhs per employee.",
        "context": "Policy Manual Section 6.1 - Insurance Benefits\n\n- Sum Insured: Rs. 5 lakhs (employee)\n- Dependent Coverage: Rs. 2 lakhs each\n- Cashless hospitalization at network hospitals\n- OPD coverage up to Rs. 25,000/year"
    },
    "benefit": {
        "answer": "Employee benefits include health insurance, gym membership reimbursement (Rs. 2,000/month), learning and development allowance (Rs. 25,000/year), and commute allowance of Rs. 3,000/month.",
        "context": "Policy Manual Section 7 - Employee Benefits\n\nBenefits include:\n1. Health Insurance (family coverage)\n2. Gym Membership: Rs. 2,000/month\n3. Learning Allowance: Rs. 25,000/year\n4. Commute Allowance: Rs. 3,000/month"
    },
    "default": {
        "answer": "I can help you with information about leave policies, work from home guidelines, insurance benefits, employee perks, and company rules. Please ask a specific question about any of these topics.",
        "context": "General Policy Information\n\nThe Employee Handbook covers:\n1. Leave Policies\n2. Work From Home Guidelines\n3. Insurance & Health Benefits\n4. Employee Benefits & Perks\n5. Code of Conduct"
    }
}

def get_demo_response(query):
    q = query.lower()
    if "leave" in q or "casual" in q or "vacation" in q:
        return DEMO_RESPONSES["leave"]
    elif "wfh" in q or "work from home" in q or "remote" in q:
        return DEMO_RESPONSES["wfh"]
    elif "insurance" in q or "health" in q or "medical" in q:
        return DEMO_RESPONSES["insurance"]
    elif "benefit" in q or "perk" in q or "allowance" in q:
        return DEMO_RESPONSES["benefit"]
    return DEMO_RESPONSES["default"]

# --------------------------------------------------
# Policy Categories
# --------------------------------------------------
CATEGORIES = [
    {"id": "leave", "name": "Leave Policy", "icon": "📅", "desc": "Vacation & time-off"},
    {"id": "wfh", "name": "Work From Home", "icon": "🏠", "desc": "Remote work guidelines"},
    {"id": "insurance", "name": "Insurance", "icon": "🛡️", "desc": "Health & life coverage"},
    {"id": "benefits", "name": "Benefits", "icon": "🎁", "desc": "Perks & allowances"},
    {"id": "rules", "name": "Company Rules", "icon": "⚖️", "desc": "Code of conduct"},
    {"id": "onboarding", "name": "Onboarding", "icon": "👋", "desc": "New employee guide"},
]

QUICK_PROMPTS = [
    "How many casual leaves do I get?",
    "What is the WFH policy?",
    "Tell me about insurance benefits",
    "What are the working hours?",
    "How do I apply for leave?"
]

# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_context" not in st.session_state:
    st.session_state.show_context = {}

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">Policy Categories</div>', unsafe_allow_html=True)

    for cat in CATEGORIES:
        st.markdown(f"""
        <div class="sidebar-category">
            <div class="sidebar-cat-icon">{cat['icon']}</div>
            <div class="sidebar-cat-text">
                <span class="sidebar-cat-name">{cat['name']}</span>
                <span class="sidebar-cat-count">{cat['desc']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Quick Questions</div>', unsafe_allow_html=True)

    for prompt in QUICK_PROMPTS:
        if st.button(prompt, key=f"quick_{prompt}", use_container_width=True):
            st.session_state.pre_filled_query = prompt
            st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="app-header">
    <h1>🏢 Employee Policy Assistant</h1>
    <p>Your intelligent guide to company policies, benefits, and HR guidelines. Ask anything about leave, WFH, insurance, and more.</p>
    <div class="header-badge">
        <div class="status-dot"></div>
        <span>AI-Powered • Ready to help</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Quick Actions
# --------------------------------------------------
st.markdown('<div class="quick-actions">', unsafe_allow_html=True)
cols = st.columns(3)
quick_actions = [
    ("📅", "Leave Policy", "Ask about time-off"),
    ("🏠", "WFH Guidelines", "Remote work rules"),
    ("🛡️", "Insurance", "Health coverage"),
]

for i, (icon, title, desc) in enumerate(quick_actions):
    with cols[i]:
        st.markdown(f"""
        <div class="quick-action-card">
            <div class="quick-action-icon">{icon}</div>
            <div class="quick-action-title">{title}</div>
            <div class="quick-action-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.markdown('<label class="input-label">What would you like to know?</label>', unsafe_allow_html=True)

# Get pre-filled query from quick prompts
default_query = st.session_state.get("pre_filled_query", "")
if "pre_filled_query" in st.session_state:
    del st.session_state["pre_filled_query"]

query = st.text_input(
    "Ask a question",
    value=default_query,
    placeholder="Example: How many casual leaves do employees get per year?",
    label_visibility="collapsed"
)

col1, col2 = st.columns([4, 1])
with col2:
    submitted = st.button("Ask →", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Answer Generation
# --------------------------------------------------
if query and submitted:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Display user message
    st.markdown(f"""
    <div class="chat-message">
        <div class="chat-avatar user">👤</div>
        <div class="chat-content">
            <div class="chat-role">You</div>
            <div class="chat-text">{query}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Loading animation
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="loading-container">
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
        <span>Searching policy documents...</span>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.5)  # Brief delay for UX

    # Check if index is available (real mode) or use demo
    if index is not None:
        # Generate query embedding
        q_emb = embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, k=3)
        context_chunks = []
        for idx in I[0]:
            if idx >= 0:
                context_chunks.append(mapping[idx])
        context = "\n\n".join(context_chunks)

        # Generate answer
        prompt = f"""
You are an HR Policy Assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, say:
"I could not find that information in the policy documents."
Context:
{context}
Question:
{query}
Answer:
"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = llm.generate(**inputs, max_new_tokens=150, temperature=0.3, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Demo mode
        demo = get_demo_response(query)
        answer = demo["answer"]
        context = demo["context"]

    loading_placeholder.empty()

    # Display response
    st.markdown(f"""
    <div class="response-container">
        <div class="response-header">
            <div class="response-icon">🤖</div>
            <div>
                <div class="response-title">Policy Assistant</div>
            </div>
        </div>
        <div class="response-text">{answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # Add to history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Context expander
    with st.expander("📄 View Source Documents"):
        st.markdown(f"""
        <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; font-size: 0.875rem; color: #4b5563; white-space: pre-wrap;">
{context}
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# Chat History (if exists)
# --------------------------------------------------
if len(st.session_state.chat_history) > 0 and not (query and submitted):
    st.markdown("### Recent Conversation")
    for msg in st.session_state.chat_history[-6:]:  # Show last 6 messages
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message">
                <div class="chat-avatar user">👤</div>
                <div class="chat-content">
                    <div class="chat-role">You</div>
                    <div class="chat-text">{msg['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message">
                <div class="chat-avatar assistant">🤖</div>
                <div class="chat-content">
                    <div class="chat-role">Policy Assistant</div>
                    <div class="chat-text">{msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------
# Clear History Button
# --------------------------------------------------
if len(st.session_state.chat_history) > 0:
    if st.button("Clear Conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
