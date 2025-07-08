import os
from coqui_tts import TTS
from config import settings

_tts = None
def _load_tts():
    global _tts
    if _tts is None:
        _tts = TTS(settings.COQUI_TTS_MODEL, progress_bar=False, gpu=False)
    return _tts

def synthesize_text(bot_id: str, text: str) -> str:
    output_dir = os.path.join("data", "audio", bot_id)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{bot_id}_{abs(hash(text))}.wav"
    path = os.path.join(output_dir, filename)
    tts = _load_tts()
    tts.tts_to_file(text, path)
    return path
