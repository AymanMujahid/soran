import os
import aiofiles
import httpx
from bs4 import BeautifulSoup
from config import settings
from embeddings import generate_embeddings
from vector_store import add_embeddings

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

async def process_upload(file, bot_id: str):
    ensure_dir(settings.FILE_STORAGE_PATH)
    path = os.path.join(settings.FILE_STORAGE_PATH, f"{bot_id}_{file.filename}")
    async with aiofiles.open(path, "wb") as out:
        await out.write(await file.read())
    texts = await extract_text_from_file(path)
    vectors = generate_embeddings(texts)
    add_embeddings(bot_id, vectors)

async def process_url(url: str, bot_id: str):
    resp = httpx.get(url, timeout=10.0)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    texts = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    vectors = generate_embeddings(texts)
    add_embeddings(bot_id, vectors)

async def process_api(endpoint: str, api_key: str, bot_id: str):
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.get(endpoint, headers=headers, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    texts = format_api_data(data)
    vectors = generate_embeddings(texts)
    add_embeddings(bot_id, vectors)

async def extract_text_from_file(path: str) -> list[str]:
    if path.lower().endswith(".pdf"):
        return await extract_text_from_pdf(path)
    if path.lower().endswith(".docx"):
        return await extract_text_from_docx(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return await extract_text_from_excel(path)
    async with aiofiles.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [await f.read()]

async def extract_text_from_pdf(path: str) -> list[str]:
    # implement PDF text extraction (e.g., via PyMuPDF)
    return []

async def extract_text_from_docx(path: str) -> list[str]:
    # implement DOCX text extraction (e.g., via python-docx)
    return []

async def extract_text_from_excel(path: str) -> list[str]:
    # implement Excel text extraction (e.g., via pandas)
    return []

def format_api_data(data) -> list[str]:
    texts = []
    if isinstance(data, list):
        for item in data:
            texts.append(str(item))
    else:
        texts.append(str(data))
    return texts
