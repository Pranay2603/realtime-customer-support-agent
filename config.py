"""
Configuration Management Module
Centralized configuration for the Customer Support Agent
"""

import os
from dataclasses import dataclass
from typing import List
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM Configuration Settings"""
    model_name: str = "llama3.2"
    temperature: float = 0.3  # Lower for more consistent support responses
    max_tokens: int = 512
    context_window: int = 4096
    

@dataclass
class RAGConfig:
    """RAG (Retrieval Augmented Generation) Configuration"""
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_results: int = 4
    similarity_threshold: float = 0.7
    

@dataclass
class VectorStoreConfig:
    """Vector Database Configuration"""
    persist_directory: str = "./vectorstore"
    collection_name: str = "support_knowledge_base"
    

@dataclass
class AudioConfig:
    """Audio Processing Configuration"""
    whisper_model: str = "base"  # Options: tiny, base, small, medium, large
    sample_rate: int = 16000
    channels: int = 1
    

@dataclass
class ServerConfig:
    """WebSocket Server Configuration"""
    host: str = "0.0.0.0"
    port = int(os.environ.get("PORT", 8000))
    max_connections: int = 100
    

@dataclass
class PathConfig:
    """File Path Configuration"""
    base_dir: Path = Path(__file__).parent
    knowledge_base_dir: Path = base_dir / "knowledge_base"
    logs_dir: Path = base_dir / "logs"
    audio_temp_dir: Path = base_dir / "temp_audio"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        self.knowledge_base_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.audio_temp_dir.mkdir(exist_ok=True)


@dataclass
class SupportedLanguages:
    """Supported Languages Configuration"""
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = [
                "en", "es", "fr", "de", "it", "pt", 
                "hi", "zh", "ja", "ko", "ar", "ru"
            ]
    
    def get_language_names(self) -> dict:
        """Return language code to name mapping"""
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "hi": "Hindi",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "ru": "Russian"
        }


class Config:
    """Main Configuration Class - Singleton Pattern"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.llm = LLMConfig()
        self.rag = RAGConfig()
        self.vectorstore = VectorStoreConfig()
        self.audio = AudioConfig()
        self.server = ServerConfig()
        self.paths = PathConfig()
        self.languages = SupportedLanguages()
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'Config':
        """Get singleton instance"""
        return cls()
    
    def update_from_env(self):
        """Update configuration from environment variables"""
        if os.getenv("LLM_MODEL"):
            self.llm.model_name = os.getenv("LLM_MODEL")
        
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        
        if os.getenv("SERVER_PORT"):
            self.server.port = int(os.getenv("SERVER_PORT"))
        
        if os.getenv("WHISPER_MODEL"):
            self.audio.whisper_model = os.getenv("WHISPER_MODEL")
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            assert 0 <= self.llm.temperature <= 1, "Temperature must be between 0 and 1"
            assert self.rag.chunk_size > 0, "Chunk size must be positive"
            assert self.rag.top_k_results > 0, "Top K results must be positive"
            assert 1024 <= self.server.port <= 65535, "Port must be between 1024 and 65535"
            return True
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"""
        Config(
            LLM Model: {self.llm.model_name},
            Temperature: {self.llm.temperature},
            RAG Chunks: {self.rag.chunk_size},
            Vector Store: {self.vectorstore.persist_directory},
            Server Port: {self.server.port},
            Languages: {len(self.languages.languages)} supported
        )
        """


# Global config instance
config = Config.get_instance()
config.update_from_env()