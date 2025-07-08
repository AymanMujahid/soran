from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import settings
import torch

_tokenizer = None
_model = None

def _load_classifier():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        _model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
    return _tokenizer, _model

def classify_chunk(text: str) -> str:
    tokenizer, model = _load_classifier()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    label_idx = torch.argmax(probs, dim=1).item()
    return model.config.id2label[label_idx]
