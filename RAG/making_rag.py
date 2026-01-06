from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
texts = []
embeddings = model.encode(texts)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype(np.float32))
print(f"Total vectors in Faiss index: {index.ntotal}")

# 벡터 DB 옵션 1: Faiss

