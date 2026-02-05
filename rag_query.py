import faiss
import json
import os
import numpy as np
import streamlit as st
import ollama
from embeddings import embed_texts

@st.cache_resource
def load_resources():
    index = faiss.read_index("employee_rag.index")
    with open("documents.json") as f:
        documents = json.load(f)
    return index, documents

index, documents = load_resources()

def rag_query(question, top_k=4):
    embed = embed_texts([question])
    distances, indices = index.search(embed, top_k)
    context = "\n\n".join([documents[i] for i in indices[0]])
    prompt = f"""
You are an HR analytics assistant.
Context:
{context}
Question:
{question}
Answer concisely and accurately.
"""
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

st.title("HR Analytics Assistant")

question = st.text_input("Ask a question about employee policies or data:", "Which employees have frequent missing check-outs and what does policy say?")

if st.button("Query"):
    with st.spinner("Searching and generating answer..."):
        answer = rag_query(question)
        st.markdown("### Answer")
        st.write(answer)