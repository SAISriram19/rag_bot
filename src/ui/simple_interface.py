#!/usr/bin/env python3
"""
Simple Working UI for Technical Documentation RAG Bot
Clean, functional interface that actually works
"""

import gradio as gr
import sys
import os
import logging
import time
from pathlib import Path
from typing import List, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the actual RAG system components
from src.config import config
from src.services.model_manager import ModelManager
from src.services.query_handler import QueryHandler
from src.services.llm_manager import LLMManager
from src.services.document_processor import DocumentProcessor, ProcessingStatus
from src.models.data_models import QueryResponse

logger = logging.getLogger(__name__)

class SimpleRAGInterface:
    """Simple working interface for the RAG bot."""
    
    def __init__(self):
        """Initialize the interface with real backend services."""
        self.model_manager = None
        self.query_handler = None
        self.llm_manager = None
        self.document_processor = None
        self.available_models = []
        self.current_model = config.default_ollama_model
        self.processing_query = False
        self.processing_files = False
        
        # Initialize components
        try:
            print("🔧 Initializing RAG components...")
            self.model_manager = ModelManager()
            self.llm_manager = LLMManager()
            self.query_handler = QueryHandler()
            self.document_processor = DocumentProcessor()
            
            self.available_models = self.model_manager.list_ollama_models()
            if not self.available_models:
                logger.warning("No Ollama models found")
                self.available_models = [config.default_ollama_model]
            
            print(f"✅ Initialized with {len(self.available_models)} models")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.available_models = [config.default_ollama_model]

def create_interface():
    """Create the working interface."""
    
    # Initialize the RAG interface
    rag_interface = SimpleRAGInterface()
    
    # Clean, professional CSS with excellent readability
    css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        background: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Global text color fix */
    .gradio-container * {
        color: #1e293b !important;
    }
    
    /* Professional header */
    .main-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%) !important;
        color: white !important;
        padding: 2.5rem !important;
        border-radius: 12px !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.2) !important;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.75rem !important;
        font-weight: 800 !important;
        margin: 0 0 0.75rem 0 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 1.2rem !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }
    
    /* Card styling */
    .card {
        background: white !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Section headers */
    .section-header {
        color: #1e293b !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* Input styling */
    .gr-textbox, .gr-dropdown, textarea, input {
        background: white !important;
        color: #1e293b !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
        transition: border-color 0.2s ease !important;
    }
    
    .gr-textbox:focus, textarea:focus, input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        outline: none !important;
    }
    
    /* Button styling */
    .gr-button {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2) !important;
    }
    
    .gr-button:hover {
        background: #2563eb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3) !important;
    }
    
    .gr-button[variant="secondary"] {
        background: #64748b !important;
        color: white !important;
    }
    
    .gr-button[variant="secondary"]:hover {
        background: #475569 !important;
    }
    
    /* Chat styling */
    .chatbot {
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        background: white !important;
    }
    
    .message {
        color: #1e293b !important;
        background: transparent !important;
    }
    
    /* File upload area */
    .file-upload {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        text-align: center !important;
        background: #f8fafc !important;
        color: #64748b !important;
        transition: all 0.2s ease !important;
    }
    
    .file-upload:hover {
        border-color: #3b82f6 !important;
        background: #eff6ff !important;
        color: #1e40af !important;
    }
    
    /* Status messages */
    .status-success {
        background: #dcfce7 !important;
        color: #166534 !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #bbf7d0 !important;
        font-weight: 500 !important;
    }
    
    .status-error {
        background: #fef2f2 !important;
        color: #991b1b !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #fecaca !important;
        font-weight: 500 !important;
    }
    
    .status-info {
        background: #eff6ff !important;
        color: #1e40af !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #bfdbfe !important;
        font-weight: 500 !important;
    }
    
    /* Source context styling */
    .source-context {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        color: #64748b !important;
        min-height: 120px !important;
        max-height: 300px !important;
        overflow-y: auto !important;
    }
    
    /* Labels */
    label, .gr-form label, .gr-block-label {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Markdown content */
    .gr-markdown, .gr-markdown * {
        color: #1e293b !important;
        background: transparent !important;
    }
    
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    
    .gr-markdown code {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        font-family: 'Monaco', 'Menlo', monospace !important;
    }
    
    .gr-markdown pre {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
    }
    """
    
    def upload_files(files):
        """Handle file uploads with real document processing."""
        if not files:
            return '<div class="status-error">❌ No files selected. Please choose some documents to upload.</div>'
        
        if rag_interface.processing_files:
            return '<div class="status-info">⏳ Processing in progress. Please wait for current batch to complete.</div>'
        
        try:
            rag_interface.processing_files = True
            
            # Validate and process files
            valid_files = []
            file_info = []
            
            for file in files:
                if file is not None:
                    file_path = file.name
                    file_name = Path(file_path).name
                    
                    if Path(file_path).suffix.lower() in config.supported_file_types:
                        valid_files.append(file_path)
                        file_size = os.path.getsize(file_path) / 1024  # KB
                        file_info.append(f"• {file_name} ({file_size:.1f} KB)")
                    else:
                        file_info.append(f"• {file_name} (unsupported format)")
            
            if not valid_files:
                return f'<div class="status-error">❌ No valid files found.<br><br><strong>Supported formats:</strong> {", ".join(config.supported_file_types)}<br><br><strong>Files checked:</strong><br>{"<br>".join(file_info)}</div>'
            
            # Process files using the real document processor
            batch_result = rag_interface.document_processor.process_multiple_documents(
                file_paths=valid_files,
                continue_on_error=True
            )
            
            if batch_result.successful_files > 0:
                return f'''<div class="status-success">
                    <strong>✅ Successfully processed {batch_result.successful_files}/{batch_result.total_files} files!</strong><br><br>
                    
                    <strong>📊 Processing Results:</strong><br>
                    • Total chunks created: <strong>{batch_result.total_chunks}</strong><br>
                    • Processing time: <strong>{batch_result.total_processing_time:.2f}s</strong><br>
                    • Success rate: <strong>{batch_result.success_rate:.1f}%</strong><br><br>
                    
                    <strong>📁 Files processed:</strong><br>
                    {"<br>".join(file_info)}<br><br>
                    
                    🎉 <strong>Your documents are now ready for querying!</strong>
                </div>'''
            else:
                return '<div class="status-error">❌ Failed to process files. Please check file formats and try again.</div>'
                
        except Exception as e:
            logger.error(f"Error in upload_files: {e}")
            return f'<div class="status-error">❌ Processing error:<br>{str(e)}</div>'
        
        finally:
            rag_interface.processing_files = False
    
    def chat_response(message, history):
        """Handle chat messages with real AI processing."""
        if not message.strip():
            return history, "", ""
        
        if rag_interface.processing_query:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⏳ Please wait for the current query to complete."})
            return history, "", ""
        
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "🤔 Processing your question..."})
        
        try:
            rag_interface.processing_query = True
            start_time = time.time()
            
            # Process query using real QueryHandler
            if rag_interface.query_handler:
                query_response = rag_interface.query_handler.handle_query(
                    query=message,
                    model=rag_interface.current_model,
                    include_conversation_history=True
                )
                
                # Format the response
                processing_time = time.time() - start_time
                confidence_emoji = "🟢" if query_response.confidence_score > 0.7 else "🟡" if query_response.confidence_score > 0.4 else "🔴"
                
                formatted_response = f"""{query_response.answer}

---
{confidence_emoji} **Confidence:** {query_response.confidence_score:.1%} | ⏱️ **Time:** {processing_time:.2f}s | 🤖 **Model:** {query_response.model_used} | 📚 **Sources:** {len(query_response.sources)}"""
                
                # Update history with actual response
                history[-1]["content"] = formatted_response
                
                # Format source context
                if query_response.sources:
                    source_context = f'<div class="source-context"><strong>📚 Found {len(query_response.sources)} relevant sources:</strong><br><br>'
                    
                    for i, source in enumerate(query_response.sources[:3], 1):  # Show top 3 sources
                        if hasattr(source, 'chunk'):
                            source_obj = source.chunk
                        else:
                            source_obj = source
                        
                        source_name = getattr(source_obj, 'source', 'Unknown Source')
                        content = getattr(source_obj, 'content', 'No content available')
                        
                        # Truncate long content
                        if len(content) > 250:
                            content = content[:250] + "..."
                        
                        source_context += f'<strong>[{i}] {source_name}</strong><br>{content}<br><br>---<br><br>'
                    
                    source_context += '</div>'
                else:
                    source_context = '<div class="source-context">📭 No sources found for this query.</div>'
                
                return history, "", source_context
                
            else:
                history[-1]["content"] = """🤖 I'm ready to help, but the query handler is not fully initialized yet.

**This might happen if:**
• The system is still starting up
• No documents have been uploaded yet
• There are configuration issues

**Try:** Upload some documents first, then ask your question again."""
                return history, "", ""
            
        except Exception as e:
            logger.error(f"Error in chat handler: {e}")
            history[-1]["content"] = f"❌ Error processing your question: {str(e)}"
            return history, "", ""
        
        finally:
            rag_interface.processing_query = False
    
    def clear_chat():
        """Clear chat history and memory."""
        try:
            if rag_interface.query_handler and hasattr(rag_interface.query_handler, 'memory_manager'):
                rag_interface.query_handler.memory_manager.clear_memory()
                logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
        return [], "", '<div class="source-context">No sources yet. Upload documents and ask questions to see relevant excerpts here.</div>'
    
    def change_model(model_name):
        """Handle model selection change."""
        if not model_name or model_name not in rag_interface.available_models:
            return f'<div class="status-error">❌ Invalid model: {model_name}</div>'
        
        try:
            if rag_interface.llm_manager:
                success = rag_interface.llm_manager.switch_model(model_name)
                if success:
                    rag_interface.current_model = model_name
                    return f'<div class="status-success">✅ Successfully switched to: <strong>{model_name}</strong></div>'
                else:
                    return f'<div class="status-error">⚠️ Failed to switch to: {model_name}</div>'
            else:
                rag_interface.current_model = model_name
                return f'<div class="status-info">✅ Selected: <strong>{model_name}</strong><br><small>LLM manager not fully initialized</small></div>'
        except Exception as e:
            logger.error(f"Error changing model: {e}")
            return f'<div class="status-error">❌ Error switching model:<br>{str(e)}</div>'
    
    # Create the interface
    with gr.Blocks(css=css, title="🤖 AI Documentation Assistant", theme=gr.themes.Default()) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 AI Documentation Assistant</h1>
            <p>Upload your technical documents and get intelligent answers with source citations</p>
        </div>
        """)
        
        # Main content
        with gr.Row():
            # Left panel - Document upload and model selection
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.HTML('<div class="section-header">📚 Document Management</div>')
                
                file_upload = gr.File(
                    label="Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"],
                    elem_classes=["file-upload"]
                )
                
                process_btn = gr.Button("🚀 Process Documents", variant="primary", size="lg")
                
                upload_status = gr.HTML('<div class="status-info">📋 Ready to process documents. Select files above and click "Process Documents".</div>')
                
                gr.HTML('<div class="section-header">🤖 Model Selection</div>')
                
                model_dropdown = gr.Dropdown(
                    choices=rag_interface.available_models,
                    value=rag_interface.current_model,
                    label="AI Model",
                    interactive=True,
                    info="Choose the AI model for generating responses"
                )
                
                model_status = gr.HTML(f'<div class="status-info">🤖 Current Model: <strong>{rag_interface.current_model}</strong></div>')
            
            # Right panel - Chat
            with gr.Column(scale=2, elem_classes=["card"]):
                gr.HTML('<div class="section-header">💬 AI Assistant</div>')
                
                chatbot = gr.Chatbot(
                    label="",
                    height=400,
                    show_copy_button=True,
                    type="messages",
                    placeholder="👋 Hello! Upload some documents and I'll help you find information in them."
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="Ask me anything about your documents...",
                        lines=2,
                        scale=4,
                        show_label=False
                    )
                    
                    with gr.Column(scale=1):
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")
                
                gr.HTML('<div class="section-header">📚 Source Context</div>')
                source_context = gr.HTML('<div class="source-context">No sources yet. Upload documents and ask questions to see relevant excerpts here.</div>')
        
        # Event handlers
        process_btn.click(upload_files, inputs=[file_upload], outputs=[upload_status])
        model_dropdown.change(change_model, inputs=[model_dropdown], outputs=[model_status])
        send_btn.click(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, source_context])
        msg_input.submit(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input, source_context])
        clear_btn.click(clear_chat, outputs=[chatbot, msg_input, source_context])
    
    return interface

def main():
    """Launch the simple working interface."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🎯 Creating AI Documentation Assistant...")
    print("✨ Professional interface with full RAG functionality")
    print("🔧 Initializing system components...")
    
    try:
        interface = create_interface()
        
        print("🚀 Launching on http://localhost:7860")
        print("📱 Opening in your browser...")
        print("⏹️ Press Ctrl+C to stop")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            inbrowser=True,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        print("🔧 Make sure all dependencies are installed and Ollama is running")
    except KeyboardInterrupt:
        print("\n👋 Thanks for using the AI Documentation Assistant!")

if __name__ == "__main__":
    main()