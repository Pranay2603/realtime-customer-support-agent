"""
Audio Handler - Fixed (Audio is Optional)
Works without whisper/gtts installed
"""

import io
import base64
from typing import Optional, Tuple
from pathlib import Path

# Try to import audio libraries (optional)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from config import config
from logger import logger


class AudioHandler:
    """
    Audio Handler with optional dependencies
    Works even if whisper/gtts are not installed
    """
    
    def __init__(self):
        self.config = config
        self.logger = logger
        self.whisper_model = None
        self.audio_enabled = WHISPER_AVAILABLE and GTTS_AVAILABLE
        
        if not self.audio_enabled:
            self.logger.warning("Audio libraries not installed. Voice features disabled.")
            self.logger.warning("To enable: pip install openai-whisper gtts")
    
    def initialize(self) -> bool:
        """Initialize audio handler"""
        if not self.audio_enabled:
            self.logger.info("Audio Handler: Disabled (libraries not installed)")
            return True  # Return True so app continues without audio
        
        try:
            self.logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model(self.config.audio.whisper_model)
            self.logger.info("Audio Handler: Enabled")
            return True
        except Exception as e:
            self.logger.error("Failed to load Whisper model", exception=e)
            self.audio_enabled = False
            return True  # Still return True to continue without audio
    
    def transcribe_audio(self, audio_data: bytes, language: Optional[str] = None) -> Tuple[str, str]:
        """Transcribe audio to text"""
        if not self.audio_enabled or not self.whisper_model:
            self.logger.warning("Audio transcription not available")
            return "", "unknown"
        
        try:
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            result = self.whisper_model.transcribe(temp_path, language=language, fp16=False)
            Path(temp_path).unlink(missing_ok=True)
            
            return result['text'].strip(), result.get('language', 'unknown')
            
        except Exception as e:
            self.logger.error("Transcription failed", exception=e)
            return "", "unknown"
    
    def synthesize_speech(self, text: str, language: str = "en") -> Optional[bytes]:
        """Convert text to speech"""
        if not self.audio_enabled:
            self.logger.warning("Speech synthesis not available")
            return None
        
        try:
            tts = gTTS(text=text, lang=language)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            self.logger.error("Speech synthesis failed", exception=e)
            return None
    
    def audio_to_base64(self, audio_bytes: bytes) -> str:
        """Convert audio bytes to base64"""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def base64_to_audio(self, base64_string: str) -> bytes:
        """Convert base64 to audio bytes"""
        return base64.b64decode(base64_string)
    
    def cleanup_temp_files(self):
        """Cleanup temporary files"""
        pass
    
    def is_enabled(self) -> bool:
        """Check if audio is enabled"""
        return self.audio_enabled