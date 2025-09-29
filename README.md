# Technical Documentation RAG Bot

A RAG-based AI assistant for querying technical documents. This application allows users to upload documents and interact with them using local AI models, with a focus on performance and privacy.

## Features

- **Modern UI**: A clean and responsive user interface.
- **Multi-format Support**: Ingests PDF, TXT, MD, and DOCX files.
- **Local AI Models**: Utilizes Ollama for local, private AI model hosting.
- **Contextual Conversations**: Maintains conversational context for follow-up questions.
- **Source Citations**: Displays the document sources used to generate answers.
- **Performance Monitoring**: Includes a dashboard for real-time system analytics.
- **Auto-optimization**: Provides intelligent performance tuning recommendations.

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama (Install from [ollama.ai](https://ollama.ai))

### Setup
```bash
# Clone the repository
git clone https://github.com/SAISriram19/rag_bot.git
cd rag_bot

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
cp .env.example .env

# Start Ollama and pull a model
ollama serve
ollama pull llama3:latest
```

### Running the Application
```bash
# To run the main Gradio interface
python -m src.main_ui

# To run the performance monitoring dashboard
python src/ui/performance_dashboard.py
```
The main interface will be available at `http://localhost:7860`.

## How to Use

1. **Upload Documents**: Upload technical documents via the user interface.
2. **Ask Questions**: Ask questions about the document content.
3. **Get Answers**: Receive AI-generated responses with source citations.
4. **Switch Models**: Change the underlying AI model for different responses.
5. **Monitor Performance**: View system analytics on the performance dashboard.

## Configuration

The application can be configured by editing the `.env` file. The following settings can be customized:
- Model settings and Ollama configuration
- Text chunk sizes and retrieval parameters
- Memory limits and conversation history
- Performance monitoring settings
- UI preferences and port settings

## Project Structure

```
rag_bot/
├── src/
│   ├── main_ui.py             # Main Gradio interface
│   ├── ui/                    # UI components and assets
│   │   ├── style.css              # Stylesheet for the main UI
│   │   └── performance_dashboard.py # Admin dashboard
│   ├── services/              # Core services
│   │   ├── document_processor.py
│   │   ├── llm_manager.py
│   │   ├── vector_store_manager.py
│   │   ├── performance_monitor.py
│   │   ├── performance_optimizer.py
│   │   └── performance_analytics.py
│   └── models/               # Data models
├── sample_dataset/           # Demo documentation
├── app.py                   # Alternative Streamlit interface
├── requirements.txt         # Dependencies
├── .env.example            # Configuration template
└── README.md               # This file
```

## Troubleshooting

- **Ollama not found**: Ensure the `ollama serve` command is running and that you have pulled a model (e.g., `ollama pull llama3:latest`).
- **Port 7860 in use**: The application will automatically find an available port if the default is in use.
- **Import errors**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.
- **Performance issues**: Refer to the performance dashboard for optimization suggestions, consider using a smaller model (e.g., `llama3:8b`), or adjust the chunk size and retrieval limits in the `.env` file.

## Use Cases

This application is suitable for querying:
- API Documentation
- Technical Manuals
- Code Documentation
- Tutorial Guides
- README files
- Enterprise documentation

## Advanced Features

- **Performance Analytics**: Track usage patterns and response times.
- **Memory Management**: Optimized conversation history management.
- **Auto-optimization**: Automatic performance tuning based on usage.
- **Multiple Interfaces**: Support for both Gradio and Streamlit UIs.
- **Comprehensive Logging**: Detailed system and error logs.