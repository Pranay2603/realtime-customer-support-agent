"""
Main Application - Simple Version (No ChromaDB)
Works with Python 3.14
"""

import asyncio
import sys
from pathlib import Path

from config import config
from logger import logger
from rag_engine_simple import SimpleRAGEngine  
from audio_handler import AudioHandler
from websocket_server import SupportAgentServer


class CustomerSupportAgent:
    def __init__(self):
        self.logger = logger
        self.rag = None
        self.audio = None
        self.server = None
        
        print("=" * 60)
        print("Customer Support Agent - Starting")
        print(f"Python Version: {sys.version}")
        print("=" * 60)
    
    def _create_sample_kb(self):
        """Create sample knowledge base"""
        kb_file = config.paths.knowledge_base_dir / "sample_faq.txt"
        kb_file.write_text("""
Customer Support FAQ

Q: What are your business hours?
A: We're available 24/7 for customer support through this chat system. 
Our live agents are available Monday-Friday 9 AM - 6 PM EST.

Q: How do I reset my password?
A: To reset your password:
1. Go to the login page
2. Click "Forgot Password"
3. Enter your registered email address
4. Check your email for a reset link
5. Follow the link and create a new password

Q: What payment methods do you accept?
A: We accept all major credit cards (Visa, Mastercard, American Express), 
PayPal, and bank transfers for enterprise customers.

Q: How long does shipping take?
A: Shipping times vary by location:
- Standard shipping: 5-7 business days
- Express shipping: 2-3 business days
- International shipping: 10-15 business days

Q: What is your return policy?
A: We offer a 30-day money-back guarantee on all products. 
Items must be unused and in original packaging. 
Return shipping is free for defective items.

Q: How do I contact customer support?
A: You can reach us via:
- This live chat (24/7)
- Email: support@company.com
- Phone: 1-800-123-4567 (Mon-Fri 9 AM - 6 PM EST)

Q: Do you offer discounts?
A: Yes! We offer:
- 10% off for first-time customers (use code: FIRST10)
- 15% off for students (with valid ID)
- Volume discounts for bulk orders
- Seasonal sales throughout the year

Q: How do I track my order?
A: Once your order ships, you'll receive a tracking number via email.
You can track your order at: www.company.com/track

Q: Can I cancel my order?
A: Yes, you can cancel within 24 hours of placing the order.
After that, the order may have already shipped.
Contact us immediately if you need to cancel.

Q: What if my product is defective?
A: We apologize for any defects! Contact us within 30 days and we'll:
- Provide a full refund, or
- Send a replacement at no charge
- Cover return shipping costs
        """)
        self.logger.info("Created sample knowledge base")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Check for knowledge base files
            kb_files = list(config.paths.knowledge_base_dir.glob("*.txt"))
            if not kb_files:
                self.logger.info("No knowledge base found, creating sample...")
                self._create_sample_kb()
                kb_files = list(config.paths.knowledge_base_dir.glob("*.txt"))
            
            # Initialize RAG Engine (Simple version)
            self.logger.info("Initializing RAG Engine...")
            self.rag = SimpleRAGEngine()
            
            if not self.rag.initialize():
                self.logger.error("RAG Engine initialization failed")
                self.logger.error("Make sure Ollama is running!")
                return False
            
            # Ingest documents
            if kb_files:
                self.logger.info(f"Loading {len(kb_files)} knowledge base files...")
                stats = self.rag.ingest_documents([str(f) for f in kb_files])
                self.logger.info(f"✅ Loaded {stats['successful']} files, {stats['total_chunks']} chunks")
            
            # Initialize audio (optional, can fail gracefully)
            self.audio = AudioHandler()
            self.audio.initialize()
            
            # Initialize WebSocket server
            self.server = SupportAgentServer(self.rag, self.audio)
            
            # Print status
            stats = self.rag.get_statistics()
            print(f"""
{'=' * 60}
✅ System Ready!
{'=' * 60}

Configuration:
  - Model: {config.llm.model_name}
  - Documents: {stats.get('total_documents', 0)}
  - Chunks: {stats.get('total_chunks', 0)}
  - Temperature: {config.llm.temperature}

Server:
  - WebSocket: ws://{config.server.host}:{config.server.port}
  - Status: Running
  
{'=' * 60}
💡 Open client.html in your browser to start chatting!
{'=' * 60}

Press Ctrl+C to stop...
            """)
            
            return True
            
        except Exception as e:
            self.logger.error("Initialization failed", exception=e)
            return False
    
    async def run(self):
        """Run the application"""
        if not await self.initialize():
            print("\n❌ Failed to start. Check the errors above.")
            sys.exit(1)
        
        try:
            await self.server.start()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error("Server error", exception=e)
            sys.exit(1)


def main():
    """Entry point"""
    try:
        app = CustomerSupportAgent()
        asyncio.run(app.run())
    except Exception as e:
        logger.error("Application failed", exception=e)
        sys.exit(1)


if __name__ == "__main__":
    main()