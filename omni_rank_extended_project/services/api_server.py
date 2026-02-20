"""
api_server.py
-------------

This file defines a simple **FastAPI** server that exposes endpoints for retrieving
recommendations, logging events, and monitoring system status.  It can be used as the
backend for a React or Streamlit front‑end.  For demonstration purposes, the
recommendation logic uses pre‑computed embeddings from the two‑tower model and an
Approximate Nearest Neighbour search library (FAISS).

To run the server:

```bash
uvicorn services.api_server:app --reload
```

This will start a local HTTP API on port 8000.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import faiss
import os

app = FastAPI(title="Feed Ranking API")

# Load embeddings (these could be stored in a database or object store in prod)
EMBED_DIR = os.getenv('EMBED_DIR', 'models/checkpoints')
user_embeddings_path = os.path.join(EMBED_DIR, 'user_embeddings.npy')
item_embeddings_path = os.path.join(EMBED_DIR, 'item_embeddings.npy')
try:
    user_embeddings = np.load(user_embeddings_path)
    item_embeddings = np.load(item_embeddings_path)
    # Build FAISS index
    index = faiss.IndexFlatIP(item_embeddings.shape[1])
    index.add(item_embeddings)
except Exception as e:
    print(f"Warning: Could not load embeddings ({e}).  API will not serve recommendations.")
    user_embeddings = None
    item_embeddings = None
    index = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommended_posts: list[int]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(user_id: int = Query(..., description="ID of the user to recommend content for"), k: int = 10):
    if user_embeddings is None or index is None:
        raise HTTPException(status_code=500, detail="Recommendation service is not initialised")
    if user_id >= len(user_embeddings):
        raise HTTPException(status_code=404, detail="User not found")
    user_vec = user_embeddings[user_id]
    # Perform ANN search for top K items
    scores, indices = index.search(user_vec.reshape(1, -1).astype('float32'), k)
    return RecommendationResponse(user_id=user_id, recommended_posts=indices[0].tolist())


class LogEventRequest(BaseModel):
    user_id: int
    post_id: int
    event_type: str
    timestamp: str


@app.post("/log")
async def log_event(event: LogEventRequest):
    # In production: write to event queue or database
    print(f"Received event: {event.json()}")
    return {"status": "logged"}