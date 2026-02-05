import ollama
import numpy as np
from concurrent.futures import ThreadPoolExecutor

EMBED_DIM = 768

def embed_texts(texts, batch_size=10, max_workers=4):
    print(f"Ollama embedding {len(texts)} documents using {max_workers} threads")
    
    def embed_batch(batch):
        response = ollama.embed(
            model="nomic-embed-text:v1.5",
                input=batch
            )
        return response["embeddings"]
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(embed_batch, batches))
    all_embeddings = [emb for batch_result in results for emb in batch_result]
    embeddings = np.asarray(all_embeddings, dtype="float32")
    
    print("Embedding complete")
    return embeddings