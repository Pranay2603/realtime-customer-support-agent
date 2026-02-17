"""
Logging Module
Centralized logging system with structured output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
import json


class StructuredLogger:
    """
    Structured Logger with file and console output
    Implements singleton pattern for consistent logging across modules
    """
    
    _instance: Optional['StructuredLogger'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 name: str = "CustomerSupportAgent",
                 log_dir: str = "./logs",
                 level: int = logging.INFO):
        
        if self._initialized:
            return
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler()
        
        self._initialized = True
    
    def _setup_console_handler(self):
        """Setup console handler with colored output"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Custom formatter with colors
        console_format = CustomFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self):
        """Setup rotating file handler"""
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message"""
        msg = self._format_message(message, **kwargs)
        if exception:
            msg += f" | Exception: {type(exception).__name__}: {str(exception)}"
        self.logger.error(msg, exc_info=exception is not None)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        msg = self._format_message(message, **kwargs)
        if exception:
            msg += f" | Exception: {type(exception).__name__}: {str(exception)}"
        self.logger.critical(msg, exc_info=exception is not None)
    
    def log_user_interaction(self, 
                            session_id: str, 
                            user_message: str, 
                            bot_response: str,
                            language: str = "en",
                            processing_time: float = 0.0):
        """Log user interaction for analytics"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "language": language,
            "user_message": user_message[:200],  # Truncate for privacy
            "bot_response": bot_response[:200],
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        # Log to separate interactions file
        interactions_file = self.log_dir / f"interactions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(interactions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(interaction) + '\n')
        
        self.info(f"User interaction logged", 
                 session_id=session_id, 
                 language=language,
                 processing_time=f"{processing_time*1000:.2f}ms")
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with additional context"""
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {context}"
        return message
    
    @classmethod
    def get_logger(cls, name: str = "CustomerSupportAgent") -> 'StructuredLogger':
        """Get logger instance (singleton)"""
        if cls._instance is None:
            cls._instance = cls(name=name)
        return cls._instance


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


# Global logger instance
logger = StructuredLogger.get_logger()