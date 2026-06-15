import streamlit as st
import faiss
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

st.set_page_config(page_title="Policy Assistant", page_icon="🏢")

@st.cache_resource
def load_assets():
    # Load Embedding Model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load FAISS Index
    index = faiss.read_index('employee_policy_faiss.index')
    
    # Load Chunk Mapping
    with open('chunk_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
        
    # Configure 4-bit Quantization to prevent OOM
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    # Load LLM
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quant_config,
        device_map="auto"
    )
    return embed_model, index, mapping, tokenizer, llm

embed_model, index, mapping, tokenizer, llm = load_assets()

st.title("🏢 Employee Policy Assistant")
st.write("Ask questions about company leaves, WFH, or insurance.")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        q_emb = embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        
        D, I = index.search(q_emb, k=3)
        context = "\n\n".join([mapping[i] for i in I[0]])
        
        prompt = f"<|system|>\nYou are a helpful assistant. Use the context to answer.</s>\n<|user|>\nContext: {context}\nQuestion: {query}</s>\n<|assistant|>"
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
        outputs = llm.generate(**inputs, max_new_tokens=150, temperature=0.3, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
        
        st.subheader("Response")
        st.success(answer)
        with st.expander("Source Context"):
            st.info(context)
