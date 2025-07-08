import os
import time
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from vector_database import get_vector_database
from embeddings import get_embedding_service

logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    """Represents context for a chat conversation"""
    bot_id: str
    session_id: str
    user_id: Optional[str]
    conversation_history: List[Dict[str, str]]
    system_prompt: Optional[str] = None
    personality: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000

@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk"""
    text: str
    similarity: float
    metadata: Dict[str, Any]
    chunk_id: str
    distance: float

@dataclass
class ChatResponse:
    """Represents a chat response"""
    message: str
    context_used: List[RetrievalResult]
    response_time: float
    tokens_used: int
    confidence: float
    sources: List[Dict[str, Any]]

class SimpleLanguageModel:
    """Simple language model interface (placeholder for actual LLM)"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.max_context_length = 4000
        
    async def generate_response(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = None
    ) -> Tuple[str, int]:
        """Generate response using language model (placeholder implementation)"""
        
        # This is a placeholder implementation
        # In production, you would integrate with OpenAI, Anthropic, or local models
        
        full_prompt = self._build_prompt(prompt, context, system_prompt)
        
        # Simulate token counting
        estimated_tokens = len(full_prompt.split()) * 1.3
        
        # Generate a simple response (placeholder)
        if context.strip():
            response = self._generate_contextual_response(prompt, context)
        else:
            response = self._generate_general_response(prompt)
        
        # Simulate response token count
        response_tokens = len(response.split()) * 1.3
        total_tokens = int(estimated_tokens + response_tokens)
        
        return response, total_tokens
    
    def _build_prompt(self, user_message: str, context: str, system_prompt: str = None) -> str:
        """Build the complete prompt"""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if context:
            parts.append(f"Context: {context}")
        
        parts.append(f"User: {user_message}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _generate_contextual_response(self, question: str, context: str) -> str:
        """Generate response based on context (placeholder)"""
        # This is a very simple placeholder
        # In production, use actual LLM APIs
        
        key_terms = question.lower().split()
        context_lower = context.lower()
        
        relevant_sentences = []
        for sentence in context.split('.'):
            if any(term in sentence.lower() for term in key_terms):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            base_response = f"Based on the provided information, {relevant_sentences[0]}"
            if len(relevant_sentences) > 1:
                base_response += f" Additionally, {relevant_sentences[1]}"
        else:
            base_response = "I can help you with that based on the information available."
        
        return base_response
    
    def _generate_general_response(self, question: str) -> str:
        """Generate general response (placeholder)"""
        return "I'd be happy to help you with that. However, I don't have specific information about this topic in my knowledge base. Could you provide more details or try asking about something else?"

class ContextManager:
    """Manages conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._contexts = {}
    
    def get_context(self, session_id: str) -> Optional[ChatContext]:
        """Get context for a session"""
        return self._contexts.get(session_id)
    
    def create_context(
        self,
        session_id: str,
        bot_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        personality: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> ChatContext:
        """Create new chat context"""
        context = ChatContext(
            bot_id=bot_id,
            session_id=session_id,
            user_id=user_id,
            conversation_history=[],
            system_prompt=system_prompt,
            personality=personality,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self._contexts[session_id] = context
        return context
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id in self._contexts:
            context = self._contexts[session_id]
            context.conversation_history.append({
                'role': role,
                'content': content,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Keep only recent messages
            if len(context.conversation_history) > self.max_history:
                context.conversation_history = context.conversation_history[-self.max_history:]
    
    def get_conversation_context(self, session_id: str, max_messages: int = 5) -> str:
        """Get recent conversation as context string"""
        if session_id not in self._contexts:
            return ""
        
        context = self._contexts[session_id]
        recent_messages = context.conversation_history[-max_messages:]
        
        conversation_text = []
        for msg in recent_messages:
            conversation_text.append(f"{msg['role']}: {msg['content']}")
        
        return "\n".join(conversation_text)
    
    def clear_context(self, session_id: str):
        """Clear context for a session"""
        if session_id in self._contexts:
            del self._contexts[session_id]

class RAGRetriever:
    """Handles document retrieval for RAG"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embedding_service = get_embedding_service(embedding_model)
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        bot_id: str,
        k: int = 5,
        similarity_threshold: float = 0.7,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve relevant document chunks for a query"""
        try:
            # Get vector database for bot
            vector_db = get_vector_database(bot_id, self.embedding_service.get_dimension())
            
            if vector_db.index.ntotal == 0:
                logger.warning(f"No documents in vector database for bot {bot_id}")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.encode_text(query)
            
            # Search for similar chunks
            search_results = vector_db.search(
                query_embedding,
                k=k * 2,  # Get more results for potential reranking
                threshold=None  # Apply threshold later
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                if result['similarity'] >= similarity_threshold:
                    retrieval_result = RetrievalResult(
                        text=result['text'],
                        similarity=result['similarity'],
                        metadata=result['metadata'],
                        chunk_id=result['id'],
                        distance=result['distance']
                    )
                    results.append(retrieval_result)
            
            # Re-rank if requested and we have multiple results
            if rerank and len(results) > 1:
                results = await self._rerank_results(query, results)
            
            # Return top k results
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for query '{query}': {e}")
            return []
    
    async def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank results based on query relevance"""
        try:
            # Simple re-ranking based on keyword overlap and length
            query_words = set(query.lower().split())
            
            for result in results:
                text_words = set(result.text.lower().split())
                
                # Calculate keyword overlap
                overlap = len(query_words.intersection(text_words))
                overlap_ratio = overlap / len(query_words) if query_words else 0
                
                # Prefer chunks with good keyword overlap and reasonable length
                length_penalty = 1.0
                if len(result.text) < 50:
                    length_penalty = 0.8
                elif len(result.text) > 2000:
                    length_penalty = 0.9
                
                # Combine similarity with keyword overlap
                result.similarity = (result.similarity * 0.7) + (overlap_ratio * 0.3) * length_penalty
            
            # Sort by updated similarity
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results
            
        except Exception as e:
            logger.warning(f"Error in re-ranking: {e}")
            return results

class RAGChatSystem:
    """Main RAG chat system"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.retriever = RAGRetriever(embedding_model)
        self.language_model = SimpleLanguageModel()
        self.context_manager = ContextManager()
        
        logger.info(f"RAG chat system initialized with embedding model: {embedding_model}")
    
    async def create_chat_session(
        self,
        session_id: str,
        bot_id: str,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        personality: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> ChatContext:
        """Create a new chat session"""
        context = self.context_manager.create_context(
            session_id=session_id,
            bot_id=bot_id,
            user_id=user_id,
            system_prompt=system_prompt,
            personality=personality,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"Created chat session {session_id} for bot {bot_id}")
        return context
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        use_context: bool = True,
        max_context_chunks: int = 3
    ) -> ChatResponse:
        """Process a chat message and generate response"""
        start_time = time.time()
        
        try:
            # Get session context
            context = self.context_manager.get_context(session_id)
            if not context:
                raise ValueError(f"No active session found: {session_id}")
            
            # Retrieve relevant context if enabled
            relevant_chunks = []
            if use_context:
                relevant_chunks = await self.retriever.retrieve_relevant_chunks(
                    query=user_message,
                    bot_id=context.bot_id,
                    k=max_context_chunks
                )
            
            # Build context string
            context_text = ""
            if relevant_chunks:
                context_text = "\n\n".join([chunk.text for chunk in relevant_chunks])
            
            # Get conversation history
            conversation_context = self.context_manager.get_conversation_context(session_id)
            
            # Build system prompt
            system_prompt = self._build_system_prompt(context.system_prompt, context.personality)
            
            # Combine all context
            full_context = ""
            if context_text:
                full_context += f"Relevant Information:\n{context_text}\n\n"
            if conversation_context:
                full_context += f"Recent Conversation:\n{conversation_context}\n\n"
            
            # Generate response
            response_text, tokens_used = await self.language_model.generate_response(
                prompt=user_message,
                context=full_context,
                temperature=context.temperature,
                max_tokens=context.max_tokens,
                system_prompt=system_prompt
            )
            
            # Calculate confidence based on retrieval quality
            confidence = self._calculate_confidence(relevant_chunks, user_message)
            
            # Add messages to conversation history
            self.context_manager.add_message(session_id, "user", user_message)
            self.context_manager.add_message(session_id, "assistant", response_text)
            
            # Prepare sources information
            sources = []
            for chunk in relevant_chunks:
                source_info = {
                    'text_preview': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    'similarity': chunk.similarity,
                    'metadata': chunk.metadata
                }
                sources.append(source_info)
            
            response_time = time.time() - start_time
            
            response = ChatResponse(
                message=response_text,
                context_used=relevant_chunks,
                response_time=response_time,
                tokens_used=tokens_used,
                confidence=confidence,
                sources=sources
            )
            
            logger.info(f"Chat response generated for session {session_id} in {response_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in chat for session {session_id}: {e}")
            raise
    
    def _build_system_prompt(self, system_prompt: Optional[str], personality: Optional[str]) -> str:
        """Build the system prompt"""
        base_prompt = "You are NOXUS, an intelligent AI assistant. Provide helpful, accurate, and relevant responses based on the available information."
        
        if personality:
            base_prompt += f" Your personality: {personality}"
        
        if system_prompt:
            base_prompt += f" Additional instructions: {system_prompt}"
        
        base_prompt += " Always be concise, clear, and helpful in your responses."
        
        return base_prompt
    
    def _calculate_confidence(self, chunks: List[RetrievalResult], query: str) -> float:
        """Calculate confidence score for the response"""
        if not chunks:
            return 0.3  # Low confidence without context
        
        # Average similarity of retrieved chunks
        avg_similarity = sum(chunk.similarity for chunk in chunks) / len(chunks)
        
        # Bonus for multiple relevant chunks
        chunk_bonus = min(len(chunks) * 0.1, 0.3)
        
        # Query length factor (longer queries might be more specific)
        query_factor = min(len(query.split()) * 0.02, 0.2)
        
        confidence = min(avg_similarity + chunk_bonus + query_factor, 1.0)
        return round(confidence, 2)
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat session"""
        context = self.context_manager.get_context(session_id)
        if not context:
            return None
        
        return {
            'session_id': session_id,
            'bot_id': context.bot_id,
            'user_id': context.user_id,
            'message_count': len(context.conversation_history),
            'created_at': context.conversation_history[0]['timestamp'] if context.conversation_history else None,
            'last_activity': context.conversation_history[-1]['timestamp'] if context.conversation_history else None,
            'settings': {
                'temperature': context.temperature,
                'max_tokens': context.max_tokens,
                'system_prompt': context.system_prompt,
                'personality': context.personality
            }
        }
    
    def clear_session(self, session_id: str):
        """Clear a chat session"""
        self.context_manager.clear_context(session_id)
        logger.info(f"Cleared chat session: {session_id}")

# Global RAG system instance
rag_system = None

def get_rag_system(embedding_model: str = "all-MiniLM-L6-v2") -> RAGChatSystem:
    """Get or create global RAG system"""
    global rag_system
    
    if rag_system is None:
        rag_system = RAGChatSystem(embedding_model)
        logger.info("Created global RAG chat system")
    
    return rag_system