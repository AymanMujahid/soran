import os
import subprocess
import torch
from whisper import load_model
from config import settings

def transcribe_audio(path: str) -> str:
    model = load_model(settings.WHISPER_MODEL)
    result = model.transcribe(path)
    return result['text']

def ocr_image(path: str) -> str:
    cmd = [settings.TESSERACT_CMD, path, 'stdout']
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    return output.decode('utf-8', errors='ignore')

# batch processing
def batch_extract(paths: list[str]) -> list[str]:
    texts = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in ['.mp3', '.wav', '.m4a']:
            texts.append(transcribe_audio(p))
        else:
            texts.append(ocr_image(p))
    return texts
