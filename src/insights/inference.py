"""
Compute insights like topics, image counts, interaction counts
for a single thread id
"""

import re
from typing import List, Tuple, Dict, Any, cast
import pandas as pd
from transformers import AutoTokenizer, pipeline, logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

from config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from ..utils import load_conversation


logging.set_verbosity_error()
# Translation
MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
translator = pipeline("translation", model=MODEL_NAME)

# Topic extraction
kw_model = KeyBERT(model="all-MiniLM-L6-v2")

# counting requests
sent_model = SentenceTransformer("all-MiniLM-L6-v2")
image_requests = [
    "please show me an image",
    "can you give me a picture",
    "draw a diagram",
    "i need a visual"
]
mindmap_requests = [
    "please create a mind map",
    "can you draw a concept map",
    "give me a mind map",
    "I want a brain-strom diagram"
]
image_embeds = sent_model.encode(image_requests, convert_to_tensor=True)
mindmap_embeds = sent_model.encode(mindmap_requests, convert_to_tensor=True)


def translate_text(text: str) -> str:
    """
    Translates multilingual text to English
    """
    max_tokens = int(0.8 * tokenizer.model_max_length)
    input_ids = tokenizer.encode(
        text, truncation=True, max_length=max_tokens, add_special_tokens=False
    )
    truncated = tokenizer.decode(input_ids, skip_special_tokens=True)
    out = translator(truncated)
    return out[0]['translation_text']


def extract_topics(text: str, top_n: int = 2) -> List[str]:
    """Return the top_n keyphrases (string-only) from English text."""
    kws_raw = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )
    # Cast to expected flat list of (phrase, score)
    kws = cast(List[Tuple[str, float]], kws_raw)
    # Unpack only the phrase part into a list of strings
    phrases: List[str] = [phrase for phrase, _ in kws]
    return phrases


def count_requests(text: str, thresh: float = 0.35) -> Tuple[int, int]:
    """Counts image and mindmap requests from a given text"""
    sentences = re.split(r'(?<=[?!.])\s+', text.lower())
    if not sentences:
        return 0, 0
    sent_embeds = sent_model.encode(sentences, convert_to_tensor=True)
    image_count = mindmap_count = 0
    for emb in sent_embeds:
        image_sim = util.cos_sim(emb, image_embeds).max().item()
        mindmap_sim = util.cos_sim(emb, mindmap_embeds).max().item()
        if image_sim >= thresh and image_sim > mindmap_sim:
            image_count += 1
        elif mindmap_sim >= thresh and mindmap_sim > image_sim:
            mindmap_count += 1
    return image_count, mindmap_count


def count_interactions(df: pd.DataFrame) -> int:
    """Counts the number of user messages in a thread"""
    return int((df['role'] != 'assistant').sum())


def run_insights(thread_id: str) -> Dict[str, Any]:
    """Returns all insights for a single thread id"""
    df = load_conversation(thread_id, SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if df.empty:
        raise ValueError(f"No conversation for thread {thread_id}")
    user_df = df[df["role"] != "assistant"]
    full_text = ' '.join(user_df["content"].tolist())

    translated = translate_text(full_text)
    topics = extract_topics(translated)
    imgs, mms = count_requests(translated)
    interactions = count_interactions(df)

    return {
        "thread_id": thread_id,
        "central_topics": topics,
        "image_requests": imgs,
        "mindmap_requests": mms,
        "interactions": interactions
    }


def main():
    """Connects logic"""
    thread_id = input("Enter thread id: ").strip()
    insights = run_insights(thread_id)
    for k, v in insights.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
