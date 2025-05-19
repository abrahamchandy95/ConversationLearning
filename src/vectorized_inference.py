"""
Handles vectorized inference of multiple thread_ids
leveraging faster computation
"""
from typing import List, Tuple, Dict, cast
from pathlib import Path
import re
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, pipeline, logging
)
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
from sqlalchemy import create_engine

from src.scores.data_setup import (
    ConversationPreprocessor, ConversationDataset
)
from src.scores.inference import get_target_columns
from src.sentiment.data_setup import TextTokenizer
from .utils import (
    select_device, load_score_model, load_sentiment_model, get_root
)
from .config import SUPABASE_URL, SUPABASE_SERVICE_KEY, DATABASE_URL


def encode_sentences(prompt_list: List[List[str]]) -> Tuple[torch.Tensor, ...]:
    """
    Encodes multiple groups of prompts into sentence embeddings

    Args:
        prompt_list (List[List[str]]): A list of prompts

    Returns:
        Tuple[torch.Tensor]: A tuple of tensors.
    """

    # Encode each group of prompts
    embeddings = []
    for group in prompt_list:
        group_embeddings = _sentence_model.encode(
            group, convert_to_tensor=True)
        embeddings.append(group_embeddings)

    # Return as a tuple for easy unpacking
    return tuple(embeddings)


def load_request_embeddings() -> Tuple[torch.Tensor, torch.Tensor]:
    _image_prompts = [
        "please show me an image",
        "can you give me a picture",
        "draw a diagram",
        "i need a visual"
    ]
    _mindmap_prompts = [
        "please create a mind map",
        "can you draw a concept map",
        "give me a mind map",
        "i want a brain-storm diagram"
    ]

    embeds = encode_sentences([_image_prompts, _mindmap_prompts])
    return embeds[0], embeds[1]


def aggr_user_messages(df: pd.DataFrame) -> pd.DataFrame:
    user_df = df[df['role'] != 'assistant']
    return (
        user_df
        .sort_values(['thread_id', 'created_at'])
        .groupby('thread_id', as_index=False)
        .agg(text=('content', lambda msgs: ' '.join(msgs.tolist())))
    )


def translate(texts: List[str]) -> List[str]:
    obj = _translator(texts)
    translated = []
    for item in obj:
        translated.append(item['translation_text'])
    return translated


def extract_topics(text: str, top_n: int = 2) -> List[str]:
    kws = _kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english', top_n=2
    )
    kws = cast(List[Tuple[str, float]], kws)
    return [phrase for phrase, _ in kws]


def count_requests(
    text: str,
    image_emb: torch.Tensor,
    mindmap_emb: torch.Tensor,
    thresh: float = 0.35
) -> Tuple[int, int]:
    sentences = re.split(r'(?<=[?!.])\s+', text.lower())
    (emb,) = encode_sentences([sentences])
    sim_img = util.cos_sim(emb, image_emb).max(dim=1).values
    sim_mm = util.cos_sim(emb, mindmap_emb).max(dim=1).values
    img_count = int(((sim_img >= thresh) & (sim_img > sim_mm)).sum().item())
    mm_count = int(((sim_mm >= thresh) & (sim_mm > sim_img)).sum().item())
    return img_count, mm_count


def compute_interactions(df: pd.DataFrame) -> Dict[str, int]:
    """
    Returns a dict mapping thread_id to count of user messages.
    """
    user_df = df[df['role'] != 'assistant']
    return user_df.groupby('thread_id').size().to_dict()


def compute_scores(
    df: pd.DataFrame,
    model_dir: Path,
    device: torch.device
) -> pd.DataFrame:

    data_dir = model_dir.parent / 'data'
    target_cols = get_target_columns(data_dir)
    model = load_score_model(model_dir, len(target_cols), device)
    # Preprocess conversations
    conv_df = ConversationPreprocessor(max_length=3000).\
        preprocess_conversations(df)
    ds = ConversationDataset(conv_df, target_columns=None)
    loader = torch.utils.data.DataLoader(ds, batch_size=32)

    all_scores = []
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            preds = model(input_ids, attention_mask)
            all_scores.append(preds.cpu())
    if all_scores:
        scores_tensor = torch.cat(all_scores, dim=0)
    else:
        scores_tensor = torch.empty((0, len(target_cols)))

    # Build DataFrame
    scores = scores_tensor.numpy()
    out = pd.DataFrame(scores, columns=target_cols)
    out['thread_id'] = conv_df['thread_id'].tolist()
    cols = ['thread_id'] + [c for c in target_cols]
    return out[cols]


def compute_sentiments(
    df: pd.DataFrame,
    model_dir: Path,
    device: torch.device,
    neg_thresh: float = 0.4,
    pos_thresh: float = 0.6,
    max_length: int = 256
) -> pd.DataFrame:

    user_conv = aggr_user_messages(df)
    # Build inference loader
    tokenizer = TextTokenizer()
    model = load_sentiment_model(model_dir, tokenizer, device)
    model.eval()
    enc = tokenizer.hf_tokenizer(
        user_conv['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    labels = [
        'negative' if p < neg_thresh else 'positive'
        if p > pos_thresh else 'neutral' for p in probs
    ]

    return pd.DataFrame({
        'thread_id': user_conv['thread_id'],
        'sentiment_label': labels,
        'p_positive': probs
    })


def compute_other_insights(df: pd.DataFrame) -> pd.DataFrame:

    user_conv = aggr_user_messages(df)
    # translate
    user_conv['translation'] = translate(user_conv['text'].tolist())
    # extract topic
    user_conv['central_topics'] = user_conv['translation'].apply(
        extract_topics)

    # get image and mindmap requests
    image_emb, mindmap_emb = load_request_embeddings()
    requests = user_conv['translation'].apply(
        lambda t: count_requests(
            text=t,
            image_emb=image_emb,
            mindmap_emb=mindmap_emb
        )
    )
    user_conv[['image_requests', 'mindmap_requests']] = pd.DataFrame(
        requests.tolist(), index=user_conv.index
    )
    # map interactions
    interactions = compute_interactions(df)
    user_conv['interactions'] = user_conv['thread_id'].\
        map(interactions).fillna(0).astype(int)

    cols = [
        'thread_id',
        'central_topics',
        'image_requests',
        'mindmap_requests',
        'interactions'
    ]
    return user_conv[cols]


if __name__ == "__main__":
    logging.set_verbosity_error()

    # Pretrained models
    TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    _sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)
    _kw_model = KeyBERT(model=SENTENCE_MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    _translator = pipeline("translation", model=TRANSLATION_MODEL_NAME)
    model_dir = get_root() / "models"
    device = select_device()

    # fetch all distinct thread_ids from 'chat messages' table
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    msg_resp = supabase.table("chat_messages").select("thread_id").execute()
    msg_ids = {r["thread_id"] for r in msg_resp.data}

    # try to fetch existing inference data from supabase
    try:
        inf_resp = supabase.table(
            "chat_inference").select("thread_id").execute()
        inf_ids = {r["thread_id"] for r in inf_resp.data}
    except Exception:
        inf_ids = set()
    # find untouched thread_ids
    to_do = msg_ids - inf_ids
    if not to_do:
        print("No new thread ids to process")
        exit(0)
    # fetch full messages from thread ids
    data_resp = supabase.table("chat_messages").select("*")\
        .in_("thread_id", list(to_do)).execute()
    resp_df = pd.DataFrame(data_resp.data)

    # Run inferences
    scores_df = compute_scores(resp_df, model_dir, device)
    sentiments_df = compute_sentiments(resp_df, model_dir, device)
    insights_df = compute_other_insights(resp_df)

    # merge all dfs into one
    merged = (
        scores_df
        .merge(sentiments_df, on="thread_id")
        .merge(insights_df,   on="thread_id")
    )
    print(merged.head())
    if not inf_ids:
        engine = create_engine(DATABASE_URL)
        merged.to_sql("chat_inference", engine,
                      if_exists="replace", index=False)
        print(f"Created and inserted {
              len(merged)} rows into 'chat_inference'.")
    else:
        records = merged.to_dict(orient="records")
        supabase.table("chat_inference").insert(records).execute()
        print(f"Appended {len(merged)} rows to 'chat_inference'.")
