import os
import hashlib
import asyncio
import aiofiles
import mimetypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
import PyPDF2
import pytesseract
from PIL import Image
import whisper
from sqlalchemy.orm import Session

from config import settings
from models import BotFile, KnowledgeChunk, FileStatus
from auth import sanitize_filename
import logging

logger = logging.getLogger(__name__)

class FileTypeError(Exception):
    pass

class FileSizeError(Exception):
    pass

class ProcessingError(Exception):
    pass

class DataProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_types = settings.ALLOWED_FILE_TYPES
        self.storage_path = settings.FILE_STORAGE_PATH
        
    def _validate_file(self, file_content: bytes, filename: str) -> Tuple[str, str]:
        file_size = len(file_content)
        if file_size > self.max_file_size:
            raise FileSizeError(f"File size {file_size} exceeds maximum {self.max_file_size}")
        
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_types:
            raise FileTypeError(f"File type {file_ext} not allowed")
        
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        return file_ext, mime_type
    
    def _generate_file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
    
    def _generate_safe_filename(self, bot_id: str, original_filename: str) -> str:
        clean_filename = sanitize_filename(original_filename)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{bot_id}_{timestamp}_{clean_filename}"
    
    async def save_file(self, bot_id: str, file_content: bytes, original_filename: str) -> BotFile:
        try:
            file_ext, mime_type = self._validate_file(file_content, original_filename)
            
            safe_filename = self._generate_safe_filename(bot_id, original_filename)
            file_path = self.storage_path / safe_filename
            
            os.makedirs(self.storage_path, exist_ok=True)
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            bot_file = BotFile(
                bot_id=bot_id,
                filename=safe_filename,
                original_filename=original_filename,
                file_path=str(file_path),
                file_size=len(file_content),
                file_type=file_ext,
                mime_type=mime_type,
                status=FileStatus.PENDING
            )
            
            self.db.add(bot_file)
            self.db.commit()
            self.db.refresh(bot_file)
            
            logger.info(f"File saved: {safe_filename} for bot {bot_id}")
            return bot_file
            
        except Exception as e:
            logger.error(f"Error saving file {original_filename}: {e}")
            raise ProcessingError(f"Failed to save file: {e}")
    
    async def process_file(self, bot_file: BotFile) -> List[str]:
        try:
            bot_file.status = FileStatus.PROCESSING
            self.db.commit()
            
            file_path = Path(bot_file.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            text_chunks = []
            
            if bot_file.file_type == '.pdf':
                text_chunks = await self._extract_from_pdf(file_path)
            elif bot_file.file_type == '.docx':
                text_chunks = await self._extract_from_docx(file_path)
            elif bot_file.file_type in ['.xlsx', '.xls']:
                text_chunks = await self._extract_from_excel(file_path)
            elif bot_file.file_type == '.txt':
                text_chunks = await self._extract_from_text(file_path)
            elif bot_file.file_type in ['.jpg', '.jpeg', '.png', '.bmp']:
                text_chunks = await self._extract_from_image(file_path)
            elif bot_file.file_type in ['.mp3', '.wav', '.m4a']:
                text_chunks = await self._extract_from_audio(file_path)
            else:
                raise ProcessingError(f"Unsupported file type: {bot_file.file_type}")
            
            if not text_chunks:
                raise ProcessingError("No text extracted from file")
            
            bot_file.status = FileStatus.COMPLETED
            bot_file.processed_at = datetime.utcnow()
            bot_file.extracted_text_length = sum(len(chunk) for chunk in text_chunks)
            bot_file.chunk_count = len(text_chunks)
            
            self.db.commit()
            
            logger.info(f"File processed successfully: {bot_file.filename}")
            return text_chunks
            
        except Exception as e:
            bot_file.status = FileStatus.FAILED
            bot_file.error_message = str(e)
            self.db.commit()
            
            logger.error(f"Error processing file {bot_file.filename}: {e}")
            raise ProcessingError(f"Failed to process file: {e}")
    
    async def _extract_from_pdf(self, file_path: Path) -> List[str]:
        try:
            text_chunks = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            chunks = self._split_text_into_chunks(text, max_chunk_size=1000)
                            text_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        continue
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ProcessingError(f"PDF extraction failed: {e}")
    
    async def _extract_from_docx(self, file_path: Path) -> List[str]:
        try:
            doc = Document(file_path)
            text_chunks = []
            
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            if full_text:
                combined_text = '\n'.join(full_text)
                text_chunks = self._split_text_into_chunks(combined_text, max_chunk_size=1000)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ProcessingError(f"DOCX extraction failed: {e}")
    
    async def _extract_from_excel(self, file_path: Path) -> List[str]:
        try:
            text_chunks = []
            
            df = pd.read_excel(file_path, sheet_name=None)
            
            for sheet_name, sheet_df in df.items():
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += sheet_df.to_string(index=False)
                
                chunks = self._split_text_into_chunks(sheet_text, max_chunk_size=1500)
                text_chunks.extend(chunks)
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Excel extraction error: {e}")
            raise ProcessingError(f"Excel extraction failed: {e}")
    
    async def _extract_from_text(self, file_path: Path) -> List[str]:
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
            
            return self._split_text_into_chunks(content, max_chunk_size=1000)
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise ProcessingError(f"Text extraction failed: {e}")
    
    async def _extract_from_image(self, file_path: Path) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
            
            def extract_text():
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image, config='--psm 6')
                return text
            
            text = await loop.run_in_executor(None, extract_text)
            
            if text.strip():
                return self._split_text_into_chunks(text, max_chunk_size=1000)
            
            return []
            
        except Exception as e:
            logger.error(f"Image OCR error: {e}")
            raise ProcessingError(f"Image OCR failed: {e}")
    
    async def _extract_from_audio(self, file_path: Path) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
            
            def transcribe_audio():
                model = whisper.load_model(settings.WHISPER_MODEL.split('/')[-1])
                result = model.transcribe(str(file_path))
                return result['text']
            
            text = await loop.run_in_executor(None, transcribe_audio)
            
            if text.strip():
                return self._split_text_into_chunks(text, max_chunk_size=1000)
            
            return []
            
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            raise ProcessingError(f"Audio transcription failed: {e}")
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            chunk_end = text.rfind(' ', start, end)
            if chunk_end == -1 or chunk_end <= start:
                chunk_end = end
            
            chunks.append(text[start:chunk_end])
            start = chunk_end - overlap
            
            if start < 0:
                start = chunk_end
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]

class URLProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_url(self, bot_id: str, url: str) -> List[str]:
        try:
            response = await self.client.get(url, follow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            
            if 'text/html' in content_type:
                return await self._extract_from_html(response.text)
            elif 'application/json' in content_type:
                return await self._extract_from_json(response.json())
            elif 'text/plain' in content_type:
                return self._split_text_into_chunks(response.text)
            else:
                raise ProcessingError(f"Unsupported content type: {content_type}")
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error processing URL {url}: {e}")
            raise ProcessingError(f"Failed to fetch URL: {e}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise ProcessingError(f"URL processing failed: {e}")
        finally:
            await self.client.aclose()
    
    async def _extract_from_html(self, html_content: str) -> List[str]:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            text_elements = []
            
            for element in soup.find_all(['p', 'div', 'article', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = element.get_text(strip=True)
                if text and len(text) > 50:
                    text_elements.append(text)
            
            if text_elements:
                combined_text = '\n'.join(text_elements)
                return self._split_text_into_chunks(combined_text)
            
            return []
            
        except Exception as e:
            logger.error(f"HTML extraction error: {e}")
            raise ProcessingError(f"HTML extraction failed: {e}")
    
    async def _extract_from_json(self, json_data: Any) -> List[str]:
        try:
            text_chunks = []
            
            def extract_text_from_json(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        extract_text_from_json(value, new_prefix)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                        extract_text_from_json(item, new_prefix)
                elif isinstance(obj, str) and len(obj) > 10:
                    text_chunks.append(f"{prefix}: {obj}")
            
            extract_text_from_json(json_data)
            
            if text_chunks:
                combined_text = '\n'.join(text_chunks)
                return self._split_text_into_chunks(combined_text)
            
            return []
            
        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            raise ProcessingError(f"JSON extraction failed: {e}")
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        return DataProcessor(self.db)._split_text_into_chunks(text, max_chunk_size)

class APIProcessor:
    def __init__(self, db: Session):
        self.db = db
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def process_api(self, bot_id: str, endpoint: str, api_key: str, headers: Optional[Dict] = None) -> List[str]:
        try:
            request_headers = {"Authorization": f"Bearer {api_key}"}
            if headers:
                request_headers.update(headers)
            
            response = await self.client.get(endpoint, headers=request_headers)
            response.raise_for_status()
            
            data = response.json()
            return await self._format_api_data(data)
            
        except httpx.HTTPError as e:
            logger.error(f"API error processing endpoint {endpoint}: {e}")
            raise ProcessingError(f"Failed to fetch API data: {e}")
        except Exception as e:
            logger.error(f"Error processing API {endpoint}: {e}")
            raise ProcessingError(f"API processing failed: {e}")
        finally:
            await self.client.aclose()
    
    async def _format_api_data(self, data: Any) -> List[str]:
        try:
            text_chunks = []
            
            if isinstance(data, list):
                for item in data:
                    text_chunks.append(str(item))
            elif isinstance(data, dict):
                for key, value in data.items():
                    text_chunks.append(f"{key}: {value}")
            else:
                text_chunks.append(str(data))
            
            combined_text = '\n'.join(text_chunks)
            return self._split_text_into_chunks(combined_text)
            
        except Exception as e:
            logger.error(f"API data formatting error: {e}")
            raise ProcessingError(f"API data formatting failed: {e}")
    
    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        return DataProcessor(self.db)._split_text_into_chunks(text, max_chunk_size)