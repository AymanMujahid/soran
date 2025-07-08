from transformers import AutoTokenizer, AutoModel
import torch
from config import settings

_tokenizer_bge = None
_model_bge = None
_tokenizer_e5 = None
_model_e5 = None

def _load_bge():
    global _tokenizer_bge, _model_bge
    if _tokenizer_bge is None or _model_bge is None:
        _tokenizer_bge = AutoTokenizer.from_pretrained(settings.BGE_MODEL)
        _model_bge = AutoModel.from_pretrained(settings.BGE_MODEL)
    return _tokenizer_bge, _model_bge

def _load_e5():
    global _tokenizer_e5, _model_e5
    if _tokenizer_e5 is None or _model_e5 is None:
        _tokenizer_e5 = AutoTokenizer.from_pretrained(settings.E5_MODEL)
        _model_e5 = AutoModel.from_pretrained(settings.E5_MODEL)
    return _tokenizer_e5, _model_e5

def generate_embeddings(texts: list[str], model_type: str = "bge") -> list[tuple[int, list[float]]]:
    tokenizer, model = _load_bge() if model_type=="bge" else _load_e5()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:,0,:].cpu().numpy().tolist()
    return list(enumerate(embeddings))
