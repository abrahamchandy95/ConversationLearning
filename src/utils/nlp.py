from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Union, Tuple, cast
import re
from torch import Tensor
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers import AutoTokenizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util


TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en"
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_REQUESTS = [
    "please show me an image",
    "can you give me a picture",
    "draw a diagram",
    "i need a visual"
]
MINDMAP_REQUESTS = [
    "please create a mind map",
    "can you draw a concept map",
    "give me a mind map",
    "I want a brain-strom diagram"
]


@dataclass
class PretrainedModels:
    translator: Pipeline = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)
    kw_model: KeyBERT = field(init=False)
    sent_model: SentenceTransformer = field(init=False)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.translator = pipeline(
                "translation", model=TRANSLATION_MODEL_NAME
            )
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                TRANSLATION_MODEL_NAME
            )
            cls._instance.sent_model = SentenceTransformer(SENTENCE_MODEL_NAME)
            cls._instance.kw_model = KeyBERT(model=SENTENCE_MODEL_NAME)
        return cls._instance


@lru_cache(maxsize=1)
def load_embedded_requests() -> Tuple[Tensor, Tensor]:
    """Returns embeddings for image and mindmap requests"""
    models = PretrainedModels()
    img_emb = models.sent_model.encode(IMAGE_REQUESTS, convert_to_tensor=True)
    mm_emb = models.sent_model.encode(MINDMAP_REQUESTS, convert_to_tensor=True)
    assert isinstance(img_emb, Tensor)
    assert isinstance(mm_emb, Tensor)
    return img_emb, mm_emb


def translate(
    text: Union[str, List],
    models: PretrainedModels = PretrainedModels()
) -> Union[str, List[str]]:
    """Translates texts into English"""
    inputs = [text] if isinstance(text, str) else text
    # tokenize in batches
    max_len = int(0.8 * models.tokenizer.model_max_length)
    encoded = models.tokenizer(
        inputs,
        truncation=True,
        max_length=max_len,
        return_tensors='pt',
        padding=True
    )
    batches = models.tokenizer.batch_decode(
        encoded["input_ids"], skip_special_tokens=True
    )
    translated = models.translator(batches)
    translated_list = [t.get("translation_text", "") for t in translated]

    if isinstance(text, str):
        return translated_list[0]
    return translated_list


def extract_topics(
    text: str,
    models: PretrainedModels = PretrainedModels(),
    top_n: int = 2
) -> List[str]:
    """Return the top_n keyphrases (string-only) from English text."""
    kws_raw = models.kw_model.extract_keywords(
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


def count_requests(
    text: str,
    models: PretrainedModels = PretrainedModels(),
    thresh: float = 0.35
) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """Counts image and mindmap requests from a given text"""
    single = isinstance(text, str)
    texts = [text] if single else text

    img_emb, mm_emb = load_embedded_requests()
    results = []
    for t in texts:
        sentences = [s.strip() for s in re.split(
            r'(?<=[?!.])\s+', t.lower()) if s.strip()]
        if not sentences:
            results.append((0, 0))
            continue

        sent_embeds = models.sent_model.encode(
            sentences, convert_to_tensor=True
        )
        assert isinstance(sent_embeds, Tensor)
        sim_img = util.cos_sim(sent_embeds, img_emb).max(dim=1).values
        sim_mm = util.cos_sim(sent_embeds, mm_emb).max(dim=1).values

        img_count = int(
            ((sim_img >= thresh) & (sim_img > sim_mm)).sum().item())
        mm_count = int(((sim_mm >= thresh) & (sim_mm > sim_img)).sum().item())
        results.append((img_count, mm_count))

    return results[0] if single else results
