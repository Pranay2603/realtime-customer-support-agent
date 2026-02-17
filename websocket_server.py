"""
WebSocket Server Module
Handles real-time bidirectional communication with clients
"""

import asyncio
import json
import uuid
from typing import Dict, Set, Optional
from datetime import datetime
import time

import websockets
from websockets.server import WebSocketServerProtocol

from config import config
from logger import logger
from rag_engine_simple import SimpleRAGEngine as RAGEngine
from audio_handler import AudioHandler


class ConnectionManager:
    """
    Manages WebSocket connections and sessions
    """
    
    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.session_metadata: Dict[str, dict] = {}
        self.logger = logger
    
    async def connect(self, websocket: WebSocketServerProtocol) -> str:
        """
        Register new connection
        
        Args:
            websocket: WebSocket connection
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        self.active_connections[session_id] = websocket
        self.session_metadata[session_id] = {
            "connected_at": datetime.now().isoformat(),
            "language": "en",
            "message_count": 0
        }
        
        self.logger.info("New connection established",
                        session_id=session_id,
                        total_connections=len(self.active_connections))
        
        return session_id
    
    async def disconnect(self, session_id: str):
        """
        Remove connection
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            
        if session_id in self.session_metadata:
            metadata = self.session_metadata[session_id]
            del self.session_metadata[session_id]
            
            self.logger.info("Connection closed",
                           session_id=session_id,
                           message_count=metadata.get("message_count", 0),
                           total_connections=len(self.active_connections))
    
    async def send_message(self, session_id: str, message: dict):
        """
        Send message to specific session
        
        Args:
            session_id: Target session
            message: Message dictionary
        """
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send(
                    json.dumps(message)
                )
            except Exception as e:
                self.logger.error("Failed to send message",
                                session_id=session_id,
                                exception=e)
    
    def get_session_language(self, session_id: str) -> str:
        """Get session language"""
        return self.session_metadata.get(session_id, {}).get("language", "en")
    
    def set_session_language(self, session_id: str, language: str):
        """Set session language"""
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["language"] = language
    
    def increment_message_count(self, session_id: str):
        """Increment message counter for session"""
        if session_id in self.session_metadata:
            self.session_metadata[session_id]["message_count"] += 1


class SupportAgentServer:
    """
    Main WebSocket Server for Customer Support Agent
    """
    
    def __init__(self, rag_engine: RAGEngine, audio_handler: AudioHandler):
        """
        Initialize server
        
        Args:
            rag_engine: RAG Engine instance
            audio_handler: Audio Handler instance
        """
        self.config = config
        self.logger = logger
        self.rag_engine = rag_engine
        self.audio_handler = audio_handler
        self.connection_manager = ConnectionManager()
        
        self.logger.info("WebSocket Server initialized")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle WebSocket connection
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        session_id = await self.connection_manager.connect(websocket)
        
        try:
            # Send welcome message
            await self._send_welcome_message(session_id)
            
            # Handle messages
            async for raw_message in websocket:
                await self._process_message(session_id, raw_message)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection closed normally", session_id=session_id)
        except Exception as e:
            self.logger.error("Connection error", session_id=session_id, exception=e)
        finally:
            await self.connection_manager.disconnect(session_id)
    
    async def _send_welcome_message(self, session_id: str):
        """Send welcome message to new connection"""
        welcome = {
            "type": "system",
            "session_id": session_id,
            "message": "Welcome to Customer Support! How can I help you today?",
            "timestamp": datetime.now().isoformat(),
            "supported_languages": list(self.config.languages.languages)
        }
        await self.connection_manager.send_message(session_id, welcome)
    
    async def _process_message(self, session_id: str, raw_message: str):
        """
        Process incoming message
        
        Args:
            session_id: Session ID
            raw_message: Raw message string
        """
        try:
            message = json.loads(raw_message)
            message_type = message.get("type", "text")
            
            self.logger.debug("Processing message",
                            session_id=session_id,
                            type=message_type)
            
            # Route based on message type
            if message_type == "text":
                await self._handle_text_message(session_id, message)
            elif message_type == "audio":
                await self._handle_audio_message(session_id, message)
            elif message_type == "language":
                await self._handle_language_change(session_id, message)
            else:
                await self._send_error(session_id, "Unknown message type")
                
        except json.JSONDecodeError:
            await self._send_error(session_id, "Invalid JSON")
        except Exception as e:
            self.logger.error("Message processing failed",
                            session_id=session_id,
                            exception=e)
            await self._send_error(session_id, "Internal server error")
    
    async def _handle_text_message(self, session_id: str, message: dict):
        """
        Handle text-based query
        
        Args:
            session_id: Session ID
            message: Message dictionary
        """
        start_time = time.time()
        
        user_message = message.get("content", "").strip()
        if not user_message:
            await self._send_error(session_id, "Empty message")
            return
        
        # Get session language
        language = self.connection_manager.get_session_language(session_id)
        
        # Send typing indicator
        await self._send_typing_indicator(session_id, True)
        
        # Query RAG engine
        answer, sources = self.rag_engine.query(user_message, language)
        
        # Stop typing indicator
        await self._send_typing_indicator(session_id, False)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Send response
        response = {
            "type": "text",
            "content": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "excerpt": doc.page_content[:150] + "..."
                }
                for doc in sources[:2]  # Limit to 2 sources
            ],
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        await self.connection_manager.send_message(session_id, response)
        
        # Log interaction
        self.logger.log_user_interaction(
            session_id=session_id,
            user_message=user_message,
            bot_response=answer,
            language=language,
            processing_time=processing_time
        )
        
        self.connection_manager.increment_message_count(session_id)
    
    async def _handle_audio_message(self, session_id: str, message: dict):
        """
        Handle audio-based query
        
        Args:
            session_id: Session ID
            message: Message dictionary with audio data
        """
        try:
            # Extract audio data
            audio_base64 = message.get("audio_data", "")
            if not audio_base64:
                await self._send_error(session_id, "No audio data")
                return
            
            # Decode audio
            audio_bytes = self.audio_handler.base64_to_audio(audio_base64)
            
            # Transcribe
            language_hint = self.connection_manager.get_session_language(session_id)
            transcribed_text, detected_language = self.audio_handler.transcribe_audio(
                audio_bytes, 
                language_hint if language_hint != "en" else None
            )
            
            if not transcribed_text:
                await self._send_error(session_id, "Could not transcribe audio")
                return
            
            # Update session language if detected differently
            if detected_language != language_hint:
                self.connection_manager.set_session_language(session_id, detected_language)
            
            # Send transcription
            await self.connection_manager.send_message(session_id, {
                "type": "transcription",
                "content": transcribed_text,
                "detected_language": detected_language
            })
            
            # Process as text query
            await self._handle_text_message(session_id, {
                "content": transcribed_text
            })
            
            # Generate audio response if requested
            if message.get("want_audio_response", False):
                await self._send_audio_response(session_id, detected_language)
                
        except Exception as e:
            self.logger.error("Audio message handling failed",
                            session_id=session_id,
                            exception=e)
            await self._send_error(session_id, "Audio processing failed")
    
    async def _send_audio_response(self, session_id: str, language: str):
        """Generate and send audio response"""
        # Get last text response (from cache or storage)
        # For simplicity, we'll skip this implementation
        pass
    
    async def _handle_language_change(self, session_id: str, message: dict):
        """
        Handle language preference change
        
        Args:
            session_id: Session ID
            message: Message with new language
        """
        new_language = message.get("language", "en")
        
        if new_language in self.config.languages.languages:
            self.connection_manager.set_session_language(session_id, new_language)
            
            await self.connection_manager.send_message(session_id, {
                "type": "system",
                "message": f"Language changed to {new_language}",
                "language": new_language
            })
            
            self.logger.info("Language changed",
                           session_id=session_id,
                           language=new_language)
        else:
            await self._send_error(session_id, "Unsupported language")
    
    async def _send_typing_indicator(self, session_id: str, is_typing: bool):
        """Send typing indicator"""
        await self.connection_manager.send_message(session_id, {
            "type": "typing",
            "is_typing": is_typing
        })
    
    async def _send_error(self, session_id: str, error_message: str):
        """Send error message"""
        await self.connection_manager.send_message(session_id, {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
    
    async def start(self):
        """Start WebSocket server"""
        self.logger.info("Starting WebSocket server",
                        host=self.config.server.host,
                        port=self.config.server.port)
        
        async with websockets.serve(
            self.handle_connection,
            self.config.server.host,
            self.config.server.port,
            max_size=10 * 1024 * 1024  # 10MB max message size
        ):
            self.logger.info("WebSocket server running",
                           host=self.config.server.host,
                           port=self.config.server.port)
            await asyncio.Future()  # Run forever