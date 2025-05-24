"""
app.py

FastAPI service for performing vectorized inference on chat message threads.

This module starts a FastAPI application with two endpoints:

  • GET /health
      Simple health check that returns {"status": "ok"}.

  • POST /inference
      Returns a list of results per thread in JSON form.

Configuration:
  The following environment variables must be set before startup:
    • SUPABASE_URL:           Your Supabase project URL
    • SUPABASE_SERVICE_KEY:   Service role key for Supabase access
    • DATABASE_URL:           Supabase URL
"""
import os
from typing import Optional, Dict, List, Any, Set
from fastapi import FastAPI
from pydantic import BaseModel
from transformers.pipelines import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from sqlalchemy import create_engine, inspect

from src.utils.utils import get_root, select_device
from src.utils.db import load_chats_from_threads
from src.complete_inference import (
    fetch_new_thread_ids, compute_inference, upsert_inference_to_supabase
)


SENTENCE_MODEL = "all-MiniLM-L6-v2"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY: str = os.environ["SUPABASE_SERVICE_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]

MODEL_DIR = get_root() / "models"
MODEL_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Conversation Scores API")


class Request(BaseModel):
    """
    List of thread IDs to process.
    If omitted, the API will fetch and new thread IDs from Supabase.
    """
    thread_ids: Optional[List[str]] = None


class Response(BaseModel):
    thread_id: str
    score: Dict[str, Any]
    sentiment: Dict[str, Any]
    insights: Dict[str, Any]


@app.on_event("startup")
def on_startup():
    app.state.supabase: Client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    )
    app.state.engine = create_engine(DATABASE_URL)
    app.state.inspector = inspect(app.state.engine)
    app.state.device = select_device()
    app.state.sentence_model = SentenceTransformer(SENTENCE_MODEL)
    app.state.keybert = KeyBERT(model=SENTENCE_MODEL)
    app.state.translator = pipeline("translation", model=TRANSLATION_MODEL)


@app.on_event("shutdown")
def on_shutdown():
    pass


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat-inference", response_model=List[Response])
def score_chats(req: Request):

    if req.thread_ids:
        tids: Set[str] = set(req.thread_ids)
    else:
        tids = fetch_new_thread_ids(app.state.supabase)
    if not tids:
        return []
    conv_df = load_chats_from_threads(app.state.supabase, list(tids))
    inference_df = compute_inference(
        conv_df, MODEL_DIR,  app.state.sentence_model, app.state.keybert, app.state.translator
    )
    upsert_inference_to_supabase(
        app.state.supabase, app.state.engine, inference_df)
    # Build response
    responses: List[Response] = []
    # Define columns belonging to sentiment and insights
    sentiment_cols = {'sentiment_label', 'p_positive'}
    insight_cols = {'central_topics', 'image_requests',
                    'mindmap_requests', 'interactions'}
    for _, row in inference_df.iterrows():
        tid = row['thread_id']
        # Score: all other columns except thread_id, sentiment, and insights
        score = {
            c: row[c]
            for c in inference_df.columns
            if c not in {'thread_id'} | sentiment_cols | insight_cols
        }
        sentiment = {
            'label': row['sentiment_label'],
            'p_positive': row['p_positive']
        }
        insights = {c: row[c] for c in insight_cols}
        responses.append(
            Response(
                thread_id=tid,
                score=score,
                sentiment=sentiment,
                insights=insights
            )
        )
    return responses
