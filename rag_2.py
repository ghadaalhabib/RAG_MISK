import os, io, textwrap
import streamlit as st
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI


# Knowledge Base (Document Store)
def read_pdf(file) -> str:
    return "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(file.read())).pages])

# Chunker
def chunk(text, size=50, overlap=20):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i+size])

# Create the database
def make_db():
    client = chromadb.Client()
    emb = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    return client, client.get_or_create_collection("docs", embedding_function=emb)

# Add documents to the database
def add_docs(col, chunks):
    ids = [f"id_{i}" for i, _ in enumerate(chunks)]
    col.add(documents=chunks, ids=ids)

#  Retriever
def search(col, q, k=4):
    return col.query(query_texts=[q], n_results=k)["documents"][0]

# Generator (LLM)
def llm_answer(question, passages):
    # Use the API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:  # fallback: extractive answer (no API)
        ctx = "\n\n".join(passages)
        return f"(No API key found in environment) Most relevant excerpt:\n\n{textwrap.shorten(ctx, width=1000)}"
    
    client = OpenAI(api_key=api_key)
    ctx = "\n\n".join(f"- {p}" for p in passages)
    prompt = f"""You are a concise assistant. Use ONLY the context to answer.

Context:
{ctx}

Question: {question}
Answer briefly and cite short quotes when helpful."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        stream=True
    )
    
    for chunk in resp:
        content = chunk.choices[0].delta.content
        if content:
            yield content

# ---------- UI ----------
st.set_page_config(page_title="Mini RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 Ask your document with RAG 🤖 ")
st.caption("Upload a PDF → ask a question → get an answer grounded in your document.")

# Integration Layer
# Upload PDF
pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if "db" not in st.session_state:
    st.session_state.db = None
    st.session_state.col = None
    st.session_state.ready = False

# Process PDF
if pdf and st.button("Process PDF"):
    with st.spinner("Reading & indexing…"):
        raw = read_pdf(pdf)
        pieces = list(chunk(raw))
        client, col = make_db()
        add_docs(col, pieces)
        st.session_state.db, st.session_state.col = client, col
        st.session_state.ready = True
    st.success(f"Indexed {len(pieces)} chunks. Ask away!")

# Ask a question
if st.session_state.ready:
    q = st.text_input("Ask a question about your PDF")
    if q:
        hits = search(st.session_state.col, q, k=4)
        
        # *** KEY CHANGE 2: Use st.write_stream to display the generator output ***
        st.markdown("### ✅ Answer")
        answer_generator = llm_answer(q, hits)
        st.write_stream(answer_generator)
        
        with st.expander("🔎 Sources (retrieved chunks)"):
            for i, h in enumerate(hits, 1):
                st.markdown(f"**Chunk {i}**\n\n{h[:800]}…")
else:
    st.info("Upload a PDF and click **Process PDF** to get started.")