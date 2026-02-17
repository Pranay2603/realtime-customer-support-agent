"""
Simple RAG Engine - In-Memory (No ChromaDB needed)
Works with Python 3.14
"""

from typing import List, Dict, Tuple
from pathlib import Path
import re

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from config import config
from logger import logger


class Document:
    """Simple document class"""
    def __init__(self, content: str, metadata: dict = None):
        self.page_content = content
        self.metadata = metadata or {}


class SimpleRAGEngine:
    """
    Simple RAG Engine using in-memory storage
    No ChromaDB required - perfect for Python 3.14
    """
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.documents = []  # Store documents in memory
        self.chunks = []     # Store chunks with metadata
        
        self.logger.info("Simple RAG Engine initialized (in-memory)")
    
    def initialize(self) -> bool:
        """Initialize the engine"""
        try:
            self.logger.info("Initializing Simple RAG Engine...")
            
            if not OLLAMA_AVAILABLE:
                self.logger.error("Ollama package not installed. Run: pip install ollama")
                return False
            
            # Test Ollama connection
            try:
                ollama.list()
                self.logger.info("Ollama connection successful")
            except Exception as e:
                self.logger.error(f"Ollama not running. Start it with: ollama serve", exception=e)
                return False
            
            self.logger.info("Simple RAG Engine ready")
            return True
            
        except Exception as e:
            self.logger.error("Initialization failed", exception=e)
            return False
    
    def ingest_documents(self, file_paths: List[str]) -> Dict:
        """Load documents into memory"""
        stats = {"successful": 0, "failed": 0, "total_chunks": 0}
        
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = self._split_text(content)
                
                for chunk in chunks:
                    self.chunks.append({
                        'content': chunk,
                        'source': Path(path).name,
                        'path': path
                    })
                
                stats["successful"] += 1
                stats["total_chunks"] += len(chunks)
                
                self.logger.info(f"Loaded {Path(path).name}", chunks=len(chunks))
                
            except Exception as e:
                stats["failed"] += 1
                self.logger.error(f"Failed to load {path}", exception=e)
        
        self.logger.info("Document ingestion complete", 
                        successful=stats["successful"],
                        total_chunks=stats["total_chunks"])
        
        return stats
    
    def _split_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into chunks"""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def _simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Simple keyword-based search"""
        query_words = set(query.lower().split())
        
        # Score each chunk
        scored_chunks = []
        for chunk in self.chunks:
            content_lower = chunk['content'].lower()
            
            # Count matching words
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top K
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for score, chunk in scored_chunks[:top_k]]
    
    def query(self, question: str, language: str = "en") -> Tuple[str, List[Document]]:
        """Query the knowledge base"""
        try:
            self.logger.debug(f"Processing query: {question[:50]}...")
            
            # Search for relevant chunks
            relevant_chunks = self._simple_search(question, top_k=3)
            
            if not relevant_chunks:
                return self._get_direct_answer(question), []
            
            # Build context
            context = "\n\n".join([chunk['content'] for chunk in relevant_chunks])
            
            # Create prompt
            prompt = f"""You are a helpful customer support agent. Use the following context to answer the question.

Context from knowledge base:
{context}

Customer Question: {question}

Instructions:
- Be polite and professional
- Answer based on the context provided
- If you don't know, say "I don't have that information in my knowledge base"
- Keep answers clear and concise

Your Response:"""
            
            # Get response from Ollama
            response = ollama.generate(
                model=self.config.llm.model_name,
                prompt=prompt,
                options={
                    'temperature': self.config.llm.temperature,
                    'num_predict': self.config.llm.max_tokens
                }
            )
            
            answer = response['response'].strip()
            
            # Convert to Document objects for compatibility
            source_docs = [
                Document(
                    content=chunk['content'],
                    metadata={'source': chunk['source']}
                )
                for chunk in relevant_chunks
            ]
            
            self.logger.debug("Query processed successfully")
            return answer, source_docs
            
        except Exception as e:
            self.logger.error("Query failed", exception=e)
            return self._get_fallback_response(language), []
    
    def _get_direct_answer(self, question: str) -> str:
        """Get answer directly from LLM without context"""
        try:
            prompt = f"""You are a helpful customer support agent. Answer this question professionally:

Question: {question}

Answer:"""
            
            response = ollama.generate(
                model=self.config.llm.model_name,
                prompt=prompt,
                options={'temperature': 0.7}
            )
            
            return response['response'].strip()
        except:
            return "I apologize, but I'm having trouble processing your request right now."
    
    def _get_fallback_response(self, language: str = "en") -> str:
        """Fallback response when query fails"""
        responses = {
            "en": "I apologize, but I'm having trouble answering that. Please try rephrasing your question.",
            "es": "Disculpe, tengo problemas para responder eso. Por favor, reformule su pregunta.",
            "fr": "Je m'excuse, j'ai du mal à répondre à cela. Veuillez reformuler votre question."
        }
        return responses.get(language, responses["en"])
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(set(c['source'] for c in self.chunks)),
            "model": self.config.llm.model_name
        }