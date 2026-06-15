import streamlit as st
import faiss
import numpy as np
import pickle
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="🏢",
    layout="wide"
)

# --------------------------------------------------
# Load Models & Assets Once
# --------------------------------------------------
@st.cache_resource
def load_assets():
    # Embedding Model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # FAISS Index
    index = faiss.read_index("employee_policy_faiss.index")

    # Chunk Mapping
    with open("chunk_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)

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
# UI
# --------------------------------------------------
st.title("🏢 Employee Policy Assistant")

st.write(
    """
    Ask questions about:
    - Leave Policy
    - Work From Home (WFH)
    - Insurance
    - Employee Benefits
    - Company Rules
    """
)

query = st.text_input(
    "Enter your question:",
    placeholder="Example: How many casual leaves do employees get per year?"
)

# --------------------------------------------------
# Answer Generation
# --------------------------------------------------
if query:

    with st.spinner("Searching policy documents..."):

        # Generate query embedding
        q_emb = embed_model.encode([query]).astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(q_emb)

        # Retrieve top chunks
        D, I = index.search(q_emb, k=3)

        context_chunks = []

        for idx in I[0]:
            if idx >= 0:
                context_chunks.append(mapping[idx])

        context = "\n\n".join(context_chunks)

    with st.spinner("Generating answer..."):

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

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=False
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    st.subheader("Response")
    st.success(answer)

    with st.expander("Retrieved Context"):
        st.write(context)
