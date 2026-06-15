import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:\\hf_cache"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# # 加载编码器
encoder = SentenceTransformer("all-MiniLM-L6-v2")
def over_similarity_obj(word1, word2, threshold=None, return_similarity=False):
    emb1 = encoder.encode(word1)
    emb2 = encoder.encode(word2)
    sim = cosine_similarity([emb1], [emb2])[0][0]

    if return_similarity:
        return sim  # 直接返回相似度
    return word2 if sim >= threshold else None

