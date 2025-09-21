from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def classify_embedding(upload_emb, custom_embeddings, custom_labels):
    """
    Classify an uploaded embedding against the stored custom embeddings.
    """
    if upload_emb is None:
        return {"class": "SIMPLE", "similarity": 0.0}

    upload_emb = upload_emb.reshape(1, -1)

    if len(custom_embeddings) == 0:
        return {"class": "SIMPLE", "similarity": 0.0}

    sims = cosine_similarity(upload_emb, custom_embeddings)[0]
    best_score = float(np.max(sims))

    if best_score > 0.9:
        match_idx = int(np.argmax(sims))
        return {
            "class": "CUSTOM",
            "similarity": best_score,
            "matched_sample": str(custom_labels[match_idx]),
        }
    else:
        return {"class": "SIMPLE", "similarity": best_score}
