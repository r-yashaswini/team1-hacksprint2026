import faiss
import json
import numpy as np
from embeddings import embed_texts
from pdf_load import load_pdf_text
with open("merged.json") as f:
    employee_docs = json.load(f)
employee_texts = [
    json.dumps(doc, ensure_ascii=False)
    for doc in employee_docs
]
pdf_text = load_pdf_text("Helix_Pro_Policy_v2.pdf")
documents = employee_texts + [pdf_text]
embeddings = embed_texts(documents)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "employee_rag.index")
with open("documents.json", "w") as f:
    json.dump(documents, f, indent=2)
print("FAISS index created")