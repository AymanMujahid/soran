# parser.py
import asyncio
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import settings

_tokenizer = None
_model = None

def _load_summarizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        _model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
    return _tokenizer, _model

async def summarize_text(text: str, max_length: int = 256) -> str:
    tokenizer, model = _load_summarizer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_length,
        min_length=64,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

async def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

async def process_document(text: str) -> list[str]:
    summary = await summarize_text(text)
    chunks = await chunk_text(summary)
    return chunks
