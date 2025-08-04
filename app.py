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
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Add src to path
sys.path.append('src')

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
    from models.data_models import QueryResponse
    from config import AppConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

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
                print(f"LLM components not available: {e}")
                self.llm_available = False
                
        except Exception as e:
            print(f"Failed to initialize components: {e}")
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
    
    def process_query(self, query: str, history: List) -> Tuple[List, str]:
        """Process user query and return response"""
        if not query.strip():
            return history, ""
        
        # Add user message to history
        history.append([query, None])
        
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
                    for i, source in enumerate(query_response.sources[:3], 1):
                        source_name = source.metadata.get('filename', 'Unknown')
                        response += f"{i}. {source_name}\n"
                
                # Add confidence and timing info
                response += f"\n⚡ *Response time: {query_response.processing_time:.2f}s*"
                
        except Exception as e:
            response = f"❌ Error processing query: {str(e)}"
        
        # Update history with response
        history[-1][1] = response
        
        return history, ""
    
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
        
        # Modern CSS styling
        css = """
        /* Modern, clean styling */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            min-height: 100vh;
        }
        
        .main-header {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 20px !important;
            padding: 30px !important;
            margin: 20px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 20px !important;
            padding: 20px !important;
            margin: 10px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(10px) !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 10px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }
        
        .status-panel {
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
            color: white !important;
            border-radius: 15px !important;
            padding: 20px !important;
            margin: 10px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
        }
        
        .gr-button {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            border-radius: 25px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        }
        
        .gr-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        }
        
        .gr-textbox, .gr-dropdown {
            border-radius: 15px !important;
            border: 2px solid rgba(102, 126, 234, 0.3) !important;
            background: rgba(255, 255, 255, 0.9) !important;
            backdrop-filter: blur(5px) !important;
        }
        
        .gr-textbox:focus, .gr-dropdown:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        .gr-file {
            border: 2px dashed #667eea !important;
            border-radius: 15px !important;
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(5px) !important;
        }
        
        .gr-chatbot {
            border-radius: 15px !important;
            background: rgba(255, 255, 255, 0.95) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        /* Message styling */
        .message {
            border-radius: 15px !important;
            padding: 15px !important;
            margin: 10px 0 !important;
            backdrop-filter: blur(5px) !important;
        }
        
        .message.user {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            margin-left: 20% !important;
        }
        
        .message.bot {
            background: rgba(255, 255, 255, 0.9) !important;
            color: #2c3e50 !important;
            margin-right: 20% !important;
            border: 1px solid rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .gradio-container > * {
            animation: fadeIn 0.6s ease-out !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-header, .chat-container, .upload-section {
                margin: 5px !important;
                padding: 15px !important;
            }
        }
        """
        
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
                outputs=[chatbot, query_input]
            )
            
            query_input.submit(
                self.process_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, query_input]
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
    print("🚀 Starting Beautiful RAG Bot...")
    
    try:
        # Initialize the app
        app = RAGBotApp()
        
        # Create and launch the interface
        interface = app.create_interface()
        
        print("✅ RAG Bot initialized successfully!")
        print("🌐 Launching web interface...")
        print("📍 URL: http://localhost:7860")
        print("⏹️ Press Ctrl+C to stop")
        
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
        print("\n👋 Shutting down RAG Bot...")
    except Exception as e:
        print(f"❌ Error starting RAG Bot: {e}")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
        print("💡 Ensure Ollama is running: ollama serve")

if __name__ == "__main__":
    main()