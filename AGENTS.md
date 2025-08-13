# Agent Instructions for rag_bot

Hello, agent! Here are some guidelines for working on this repository.

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot. It uses local LLMs via Ollama, a vector store (ChromaDB), and a Python backend built with LangChain. The user interface is created with Gradio.

The core logic is in the `src/` directory, which is divided into `services`, `ui`, and `models`. The main application entry points are `app.py` and `src/ui/simple_interface.py`.

## Development Workflow

1.  **Understand the Goal:** Read the user's request carefully. Examine the existing code to understand the context.
2.  **Create a Plan:** Before writing any code, create a clear, step-by-step plan using the `set_plan` tool.
3.  **Code implementation:**
    - All new code should be well-documented with docstrings and comments where necessary.
    - All functions should have type hints.
    - Follow the existing modular structure. For example, new core logic should likely go into a new or existing service in `src/services`.
    - If you modify any core logic, you **must** add or update tests in the `tests/` directory.
4.  **Testing:**
    - This project uses `pytest`. Before submitting, you must run the test suite to ensure your changes haven't broken anything.
    - To run the tests, use the command `pytest` in the root directory.
5.  **Documentation:**
    - If you add or change functionality, update the `README.md` accordingly.
    - Keep this `AGENTS.md` file updated if you change the development workflow or architecture in a significant way.

## Important Notes

-   **Dependencies:** All dependencies are managed in `requirements.txt`. If you add a new dependency, make sure to add it to this file.
-   **Configuration:** The application is configured through environment variables, defined in `.env.example`. Do not commit a `.env` file.
-   **UI:** The UI is built with Gradio. The CSS is in `src/ui/style.css`.
-   **Entry Points:** The main user-facing application can be started by running `python app.py` or `python src/ui/simple_interface.py`.

By following these guidelines, you'll help keep the codebase clean, maintainable, and easy to work with. Thank you!
