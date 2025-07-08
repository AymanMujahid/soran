import os
import faiss
import numpy as np
from config import settings

_index = None
_id_map = {}

def _ensure_index(dim: int):
    global _index
    if _index is None:
        index_path = settings.VECTOR_DB_PATH
        os.makedirs(index_path, exist_ok=True)
        faiss_index_file = os.path.join(index_path, "faiss.index")
        if os.path.exists(faiss_index_file):
            _index = faiss.read_index(faiss_index_file)
        else:
            _index = faiss.index_factory(dim, settings.FAISS_INDEX_FACTORY)
        _index.nprobe = 10

def save_index():
    index_path = settings.VECTOR_DB_PATH
    faiss_index_file = os.path.join(index_path, "faiss.index")
    faiss.write_index(_index, faiss_index_file)

def add_embeddings(bot_id: str, vectors: list[tuple[int, list[float]]]):
    dims = len(vectors[0][1])
    _ensure_index(dims)
    ids = []
    feats = []
    for local_id, emb in vectors:
        uid = f"{bot_id}_{local_id}"
        idx = len(_id_map)
        _id_map[uid] = idx
        ids.append(idx)
        feats.append(emb)
    feats_np = np.array(feats).astype("float32")
    _index.add_with_ids(feats_np, np.array(ids))
    save_index()

def query_embeddings(bot_id: str, top_k: int = 5, query_emb: list[float] = None):
    dims = len(query_emb)
    _ensure_index(dims)
    xb = np.array(query_emb).astype("float32").reshape(1, -1)
    distances, indices = _index.search(xb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        for uid, mapped in _id_map.items():
            if mapped == idx and uid.startswith(f"{bot_id}_"):
                results.append((uid, float(dist)))
                break
    return results
