#!/usr/bin/env python3
"""
Beautiful Technical Documentation RAG Bot
A modern, clean interface for document-based AI assistance
"""

import gradio as gr
import sys
import os
import time
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Import core services
try:
    from services.document_processor import DocumentProcessor, ProcessingStatus
    from services.query_handler import QueryHandler
    from services.vector_store_manager import VectorStoreManager
    from services.embedding_generator import EmbeddingGenerator
    from services.llm_manager import LLMManager
    from services.memory_manager import MemoryManager
    from services.retriever import Retriever
    from services.response_generator import ResponseGenerator
    from services.model_manager import ModelManager
    from services.logging_config import get_logging_manager
    from models.data_models import QueryResponse
    from config import AppConfig
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.error(f"Import error: {e}")
    logging.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)
logging_manager = get_logging_manager()


class RAGBotApp:
    """Main RAG Bot Application with beautiful UI"""
    
    def __init__(self):
        """Initialize the RAG Bot application"""
        self.config = AppConfig()
        self.conversation_history = []
        self.uploaded_files = []
        self.processing_status = {}
        
        # Initialize core components
        try:
            self.vector_store = VectorStoreManager(self.config)
            self.embedding_gen = EmbeddingGenerator(self.config)
            self.doc_processor = DocumentProcessor(
                embedding_generator=self.embedding_gen,
                vector_store_manager=self.vector_store
            )
            
            # Initialize LLM components
            try:
                self.llm_manager = LLMManager()
                self.model_manager = ModelManager()
                self.memory_manager = MemoryManager()
                self.retriever = Retriever(self.vector_store, self.embedding_gen)
                self.response_generator = ResponseGenerator(self.llm_manager)
                self.query_handler = QueryHandler(
                    retriever=self.retriever,
                    response_generator=self.response_generator,
                    memory_manager=self.memory_manager
                )
                self.llm_available = True
            except Exception as e:
                logger.warning(f"LLM components not available: {e}")
                self.llm_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise
    
    def get_system_status(self) -> str:
        """Get current system status"""
        try:
            collection_info = self.vector_store.get_collection_info()
            doc_count = collection_info.get('count', 0)
            
            if self.llm_available:
                try:
                    models = self.model_manager.get_available_models()
                    model_status = f"✅ {len(models)} models available"
                except:
                    model_status = "⚠️ LLM connection issues"
            else:
                model_status = "❌ LLM not available"
            
            status = f"""
            📊 **System Status**
            
            • **Documents**: {doc_count} processed
            • **Vector Store**: {'✅ Ready' if doc_count > 0 else '📝 Empty'}
            • **LLM Models**: {model_status}
            • **Memory**: ✅ Active
            """
            
            return status.strip()
        except Exception as e:
            return f"❌ System error: {str(e)}"
    
    def upload_documents(self, files) -> str:
        """Process uploaded documents"""
        if not files:
            return "❌ No files selected"
        
        results = []
        success_count = 0
        
        for file in files:
            try:
                file_path = file.name if hasattr(file, 'name') else str(file)
                filename = Path(file_path).name
                
                # Process the document
                result = self.doc_processor.process_document(file_path)
                
                if result.status == ProcessingStatus.COMPLETED:
                    success_count += 1
                    results.append(f"✅ **{filename}**: {result.chunks_created} chunks created")
                    self.uploaded_files.append({
                        'name': filename,
                        'path': file_path,
                        'chunks': result.chunks_created,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
                else:
                    results.append(f"❌ **{filename}**: {result.error_message or 'Processing failed'}")
                    
            except Exception as e:
                results.append(f"❌ **{Path(file_path).name}**: {str(e)}")
        
        summary = f"📁 **Upload Complete**: {success_count}/{len(files)} files processed successfully\n\n"
        return summary + "\n".join(results)
    
    def get_file_list(self) -> str:
        """Get list of uploaded files"""
        if not self.uploaded_files:
            return "📝 No files uploaded yet"
        
        file_list = ["📚 **Uploaded Documents**\n"]
        for i, file_info in enumerate(self.uploaded_files[-10:], 1):  # Show last 10 files
            file_list.append(f"{i}. **{file_info['name']}** - {file_info['chunks']} chunks ({file_info['timestamp']})")
        
        return "\n".join(file_list)
    
    def process_query(self, query: str, history: List) -> Tuple[List, str, str]:
        """Process user query and return response"""
        if not query.strip():
            return history, "", "No query entered."
        
        # Add user message to history
        history.append([query, None])
        
        source_context = "No sources found for this query."

        try:
            if not self.llm_available:
                # Simple response without LLM
                collection_info = self.vector_store.get_collection_info()
                doc_count = collection_info.get('count', 0)
                
                if doc_count == 0:
                    response = "📝 Please upload some documents first! I need documents to answer your questions."
                else:
                    response = f"📚 I can see you have {doc_count} document chunks available. However, the LLM service is not available for generating responses. Please ensure Ollama is running with a model loaded."
            else:
                # Full RAG response
                query_response = self.query_handler.handle_query(query)
                
                # Format response with sources
                response = query_response.answer
                
                if query_response.sources:
                    response += "\n\n📖 **Sources:**\n"
                    source_context_parts = ["**📚 Relevant Sources:**\n\n"]
                    for i, source in enumerate(query_response.sources[:3], 1):
                        source_name = source.metadata.get('filename', 'Unknown')
                        response += f"{i}. {source_name}\n"

                        # Truncate long content
                        content = source.page_content
                        if len(content) > 250:
                            content = content[:250] + "..."

                        source_context_parts.append(f'**[{i}] {source_name}**\n\n---\n\n{content}\n\n')
                    source_context = "".join(source_context_parts)
                
                # Add confidence and timing info
                response += f"\n⚡ *Response time: {query_response.processing_time:.2f}s*"
                
        except Exception as e:
            response = f"❌ Error processing query: {str(e)}"
        
        # Update history with response
        history[-1][1] = response
        
        return history, "", source_context
    
    def clear_conversation(self) -> Tuple[List, str]:
        """Clear conversation history"""
        self.conversation_history = []
        if hasattr(self, 'memory_manager') and self.memory_manager:
            try:
                self.memory_manager.clear_memory()
            except:
                pass
        return [], ""
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            if self.llm_available:
                return self.model_manager.get_available_models()
            else:
                return ["No models available"]
        except:
            return ["Error loading models"]
    
    def switch_model(self, model_name: str) -> str:
        """Switch to a different model"""
        try:
            if self.llm_available and model_name != "No models available":
                # Update the model in LLM manager
                self.llm_manager.current_model = model_name
                return f"✅ Switched to model: {model_name}"
            else:
                return "❌ Cannot switch model - LLM not available"
        except Exception as e:
            return f"❌ Error switching model: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the beautiful Gradio interface"""
        
        # Load CSS from file
        css_path = Path(__file__).parent / "ui/style.css"
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css = f.read()
        except FileNotFoundError:
            logger.warning("style.css not found, using default styles.")
            css = None

        # Create the interface
        with gr.Blocks(
            css=css,
            title="🤖 RAG Bot - Document AI Assistant",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="purple",
                neutral_hue="gray"
            )
        ) as interface:
            
            # Header
            with gr.Row(elem_classes=["main-header"]):
                gr.Markdown("""
                # 🤖 Technical Documentation RAG Bot
                ### Your AI-powered document assistant
                
                Upload your technical documents and ask questions about them using advanced AI models.
                """)
            
            # Main content area
            with gr.Row():
                # Left column - Chat interface
                with gr.Column(scale=2, elem_classes=["chat-container"]):
                    gr.Markdown("## 💬 Chat with your documents")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_label=False,
                        avatar_images=("👤", "🤖"),
                        bubble_full_width=False
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder="Ask me anything about your documents...",
                            label="Your question",
                            lines=2,
                            scale=4,
                            show_label=False
                        )
                        send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                        example_btn = gr.Button("Example Questions 💡", variant="secondary")
                
                # Right column - Controls and status
                with gr.Column(scale=1):
                    # System status
                    with gr.Group(elem_classes=["status-panel"]):
                        gr.Markdown("## 📊 System Status")
                        status_display = gr.Markdown(
                            value=self.get_system_status(),
                            label="Status"
                        )
                        refresh_status_btn = gr.Button("Refresh Status 🔄", variant="secondary")
                    
                    # File upload section
                    with gr.Group(elem_classes=["upload-section"]):
                        gr.Markdown("## 📁 Upload Documents")
                        
                        file_upload = gr.File(
                            label="Select files",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx"],
                            height=120
                        )
                        
                        upload_btn = gr.Button("Process Files 📤", variant="primary")
                        upload_status = gr.Markdown(
                            value="Ready to upload documents",
                            label="Upload Status"
                        )
                    
                    # Model selection
                    with gr.Group(elem_classes=["upload-section"]):
                        gr.Markdown("## 🤖 AI Model")
                        
                        model_dropdown = gr.Dropdown(
                            choices=self.get_available_models(),
                            value=self.get_available_models()[0] if self.get_available_models() else None,
                            label="Select Model",
                            interactive=True
                        )
                        
                        switch_model_btn = gr.Button("Switch Model 🔄", variant="secondary")
                        model_status = gr.Markdown(value="Model ready")
                    
                    # File list
                    with gr.Group(elem_classes=["upload-section"]):
                        gr.Markdown("## 📚 Document Library")
                        file_list_display = gr.Markdown(
                            value=self.get_file_list(),
                            label="Uploaded Files"
                        )

                    # Source context
                    with gr.Group(elem_classes=["upload-section"]):
                        gr.Markdown("## 📚 Source Context")
                        source_context_display = gr.Markdown(
                            value="Source context will appear here.",
                            label="Source Context"
                        )
            
            # Example questions section (initially hidden)
            with gr.Row(visible=False) as examples_row:
                with gr.Column():
                    gr.Markdown("## 💡 Example Questions")
                    example_questions = [
                        "What is this document about?",
                        "Can you summarize the main points?",
                        "What are the key features mentioned?",
                        "How do I get started?",
                        "What are the requirements?",
                        "Are there any code examples?"
                    ]
                    
                    for question in example_questions:
                        example_q_btn = gr.Button(f"❓ {question}", variant="secondary")
                        example_q_btn.click(
                            lambda q=question: (q, ""),
                            outputs=[query_input, gr.Textbox()]
                        )
            
            # Event handlers
            def toggle_examples():
                return gr.update(visible=not examples_row.visible)
            
            # Chat functionality
            send_btn.click(
                self.process_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, query_input, source_context_display]
            )
            
            query_input.submit(
                self.process_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, query_input, source_context_display]
            )
            
            clear_btn.click(
                self.clear_conversation,
                outputs=[chatbot, query_input]
            )
            
            example_btn.click(
                toggle_examples,
                outputs=[examples_row]
            )
            
            # File upload functionality
            upload_btn.click(
                self.upload_documents,
                inputs=[file_upload],
                outputs=[upload_status]
            ).then(
                lambda: self.get_file_list(),
                outputs=[file_list_display]
            ).then(
                lambda: self.get_system_status(),
                outputs=[status_display]
            )
            
            # Model switching
            switch_model_btn.click(
                self.switch_model,
                inputs=[model_dropdown],
                outputs=[model_status]
            )
            
            # Status refresh
            refresh_status_btn.click(
                lambda: self.get_system_status(),
                outputs=[status_display]
            )
            
            # Auto-refresh status every 30 seconds
            interface.load(
                lambda: self.get_system_status(),
                outputs=[status_display],
                every=30
            )
        
        return interface

def main():
    """Main function to run the RAG Bot"""
    logger.info("🚀 Starting Beautiful RAG Bot...")
    
    try:
        # Initialize the app
        app = RAGBotApp()
        
        # Create and launch the interface
        interface = app.create_interface()
        
        logger.info("✅ RAG Bot initialized successfully!")
        logger.info("🌐 Launching web interface...")
        logger.info("📍 URL: http://localhost:7860")
        logger.info("⏹️ Press Ctrl+C to stop")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True,
            show_tips=True,
            height=800,
            favicon_path=None
        )
        
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down RAG Bot...")
    except Exception as e:
        logger.error(f"❌ Error starting RAG Bot: {e}", exc_info=True)
        logger.error("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        logger.error("💡 Ensure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()