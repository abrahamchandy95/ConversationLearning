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
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from src.utils.db import TableFetcher
from src.utils.utils import get_root
from src.utils.device import select_device
from src.complete_inference import (
    fetch_new_thread_ids, compute_inference, upsert_inference_to_supabase
)


load_dotenv()
DATABASE_URL: str = os.environ["DATABASE_URL"]
MODEL_DIR = get_root() / "models"
MODEL_DIR.mkdir(exist_ok=True)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Missing supabase credentials!")
    supabase: Client = create_client(url, key)
    app.state.supabase = supabase
    app.state.device = select_device()
    app.state.fetcher = TableFetcher(client=supabase, table="chat_messages")

    yield

# FastAPI app
app = FastAPI(title="Conversation Scores API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


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
    # Load conversation data via the fetcher
    try:
        conv_df = app.state.fetcher.fetch_all(thread_ids=list(tids))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Run inference pipeline
    inference_df = compute_inference(conv_df, MODEL_DIR)
    upsert_inference_to_supabase(
        app.state.supabase,
        inference_df
    )

    # Assemble response
    responses: List[Response] = []
    sentiment_cols = {"sentiment_label", "p_positive"}
    insight_cols = {"central_topics", "image_requests",
                    "mindmap_requests", "interactions"}

    for _, row in inference_df.iterrows():
        tid = row["thread_id"]
        score = {c: row[c] for c in inference_df.columns if c not in (
            {"thread_id"} | sentiment_cols | insight_cols)}
        sentiment = {"label": row["sentiment_label"],
                     "p_positive": row["p_positive"]}
        insights = {c: row[c] for c in insight_cols}
        responses.append(Response(thread_id=tid, score=score,
                         sentiment=sentiment, insights=insights))

    return responses
