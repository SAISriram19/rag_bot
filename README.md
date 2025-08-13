# 🤖 Technical Documentation RAG Bot

A beautiful, modern AI assistant for your technical documents. Upload documents and chat with them using local AI models with comprehensive performance monitoring.

## ✨ Features

- 🎨 **Beautiful Modern UI** - Clean, responsive interface with smooth animations
- 📚 **Multi-format Support** - PDF, TXT, MD, DOCX files
- 🤖 **Local AI Models** - Uses Ollama for complete privacy
- 💬 **Smart Conversations** - Maintains context across questions
- � **Smource Citations** - Shows which documents were used for answers
- ⚡ **Real-time Performance Monitoring** - Live system analytics and optimization
- 📊 **Performance Dashboard** - Admin interface with detailed metrics
- 🔧 **Auto-optimization** - Intelligent performance tuning

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **Ollama** - Install from [ollama.ai](https://ollama.ai)

### 2. Setup
```bash
# Clone and enter directory
git clone https://github.com/SAISriram19/rag_bot.git
cd rag_bot

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Start Ollama and pull a model
ollama serve
ollama pull llama3:latest
```

### 3. Run the App
```bash
# Main Gradio interface
# Run from the root directory
python -m src.main_ui

# Performance monitoring dashboard
python src/ui/performance_dashboard.py
```

The beautiful interface will open at `http://localhost:7860` 🎉

## 📖 How to Use

1. **Upload Documents** 📁 - Drag and drop your technical docs
2. **Ask Questions** 💬 - Chat naturally about your documents  
3. **Get Smart Answers** 🧠 - Receive AI responses with source citations
4. **Switch Models** 🔄 - Try different AI models for varied responses
5. **Monitor Performance** 📊 - View real-time analytics and optimization suggestions

## 🛠️ Configuration

Edit `.env` file to customize:
- Model settings and Ollama configuration
- Chunk sizes and retrieval parameters
- Memory limits and conversation history
- Performance monitoring settings
- UI preferences and port settings

## 📁 Project Structure

```
rag_bot/
├── src/
│   ├── main_ui.py             # 🎨 Main Gradio interface
│   ├── ui/                    # 🎨 UI components and assets
│   │   ├── style.css              # Stylesheet for the main UI
│   │   └── performance_dashboard.py # Admin dashboard
│   ├── services/              # 🔧 Core services
│   │   ├── document_processor.py
│   │   ├── llm_manager.py
│   │   ├── vector_store_manager.py
│   │   ├── performance_monitor.py
│   │   ├── performance_optimizer.py
│   │   └── performance_analytics.py
│   └── models/               # 📊 Data models
├── sample_dataset/           # 📚 Demo documentation
├── app.py                   # 🎨 Alternative Streamlit interface
├── requirements.txt         # 📦 Dependencies
├── .env.example            # ⚙️ Configuration template
└── README.md               # 📖 This file
```

## 🔧 Troubleshooting

**Ollama not found?**
```bash
ollama serve
ollama pull llama3:latest
```

**Port 7860 in use?**
```bash
# The app will automatically find an available port
```

**Import errors?**
```bash
pip install -r requirements.txt
```

**Performance issues?**
- Check the performance dashboard for optimization suggestions
- Try smaller models like `llama3:8b` for faster responses
- Reduce chunk size and retrieval limits in `.env`

## 🎯 Perfect For

- 📋 API Documentation
- 📚 Technical Manuals  
- 🔍 Code Documentation
- 📖 Tutorial Guides
- 📝 README files
- 🏢 Enterprise documentation systems

## 🚀 Advanced Features

- **Performance Analytics** - Track usage patterns and response times
- **Memory Management** - Intelligent conversation history optimization
- **Auto-optimization** - Automatic performance tuning based on usage
- **Multiple Interfaces** - Choose between Gradio and Streamlit UIs
- **Comprehensive Logging** - Detailed system and error logging

---

**Made with ❤️ for developers who love beautiful,functional AI tools with enterprise-grade performance monitoring**
