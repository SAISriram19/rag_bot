# 🤖 Technical Documentation RAG Bot

A beautiful, modern AI assistant for your technical documents. Upload documents and chat with them using local AI models.

## ✨ Features

- 🎨 **Beautiful Modern UI** - Clean, responsive interface with smooth animations
- 📚 **Multi-format Support** - PDF, TXT, MD, DOCX files
- 🤖 **Local AI Models** - Uses Ollama for complete privacy
- 💬 **Smart Conversations** - Maintains context across questions
- 📖 **Source Citations** - Shows which documents were used for answers
- ⚡ **Real-time Status** - Live system monitoring and file management

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.8+**
- **Ollama** - Install from [ollama.ai](https://ollama.ai)

### 2. Setup
```bash
# Clone and enter directory
git clone <your-repo>
cd technical-doc-rag-bot

# Install dependencies
pip install -r requirements.txt

# Start Ollama and pull a model
ollama serve
ollama pull llama3:latest
```

### 3. Run the App
```bash
python app.py
```

The beautiful interface will open at `http://localhost:7860` 🎉

## 📖 How to Use

1. **Upload Documents** 📁 - Drag and drop your technical docs
2. **Ask Questions** 💬 - Chat naturally about your documents  
3. **Get Smart Answers** 🧠 - Receive AI responses with source citations
4. **Switch Models** 🔄 - Try different AI models for varied responses

## 🛠️ Configuration

Edit `src/config.py` to customize:
- Model settings
- Chunk sizes
- Memory limits
- UI preferences

## 📁 Project Structure

```
technical-doc-rag-bot/
├── app.py              # 🎨 Beautiful main application
├── src/                # 🔧 Core services and models
├── requirements.txt    # 📦 Dependencies
└── README.md          # 📖 This file
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

## 🎯 Perfect For

- 📋 API Documentation
- 📚 Technical Manuals  
- 🔍 Code Documentation
- 📖 Tutorial Guides
- 📝 README files

---

**Made with ❤️ for developers who love beautiful, functional AI tools**
