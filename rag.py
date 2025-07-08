import os
import json
from typing import List, Dict
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from vector_store import query_embeddings
from embeddings import generate_embeddings
from config import settings

_prompt_template = """
You are NOXUS, the central AI core. Use the following context to answer the question.
{context}

Question: {question}
Answer in a concise, authoritative tone.
"""

prompt = PromptTemplate(template=_prompt_template, input_variables=["context", "question"])

def _retrieve_context(bot_id: str, question: str, k: int = 5) -> str:
    query_emb = generate_embeddings([question], model_type="e5")[0][1]
    results = query_embeddings(bot_id, top_k=k, query_emb=query_emb)
    docs: List[str] = []
    for uid, _ in results:
        file_id, chunk_id = uid.split("_", 1)
        # load chunk text from storage
        path = os.path.join(settings.VECTOR_DB_PATH, f"{uid}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                docs.append(data.get("text", ""))
    return "\n\n".join(docs)

async def chat_with_bot(bot_id: str, question: str, session: List[Dict]) -> str:
    context = _retrieve_context(bot_id, question)
    full_context = "\n\n".join([f"User: {m['user']}\nBot: {m['bot']}" for m in session])
    combined = f"{full_context}\n\n{context}"
    llm = OpenAI(model_name=settings.LLM_MODEL, temperature=0.0)
    qa = RetrievalQA(llm=llm, prompt=prompt, retriever=None)
    answer = qa.run({"context": combined, "question": question})
    return answer
