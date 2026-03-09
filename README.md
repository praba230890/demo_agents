# LangChain Toy Agent (Single Agent, Local Ollama)

This project is a minimal learning setup for agent architecture using:
- Local model: `llama3.1:8b` via Ollama
- Framework: LangChain
- Pattern: Single tool-calling agent

## 1) Prerequisites

1. Start Ollama server (usually automatic once Ollama is installed).
2. Ensure your model exists:
   - `ollama pull llama3.1:8b`
3. Create a Python env and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
   - `pip install -r requirements.txt`

## 2) Run

- `python single_agent.py`

Try prompts like:
- `What is (24 * 7) + sqrt(81)?`
- `How many words are in: langchain agents are graph-like control loops`
- `Explain in one paragraph what an agent is.`

## 3) What this teaches (architecture)

The script has these components:
1. **Model** (`ChatOllama`): reasoning engine.
2. **Tools** (`calculate`, `word_count`): external capabilities.
3. **Prompt**: behavior contract (when to use tools).
4. **Agent** (`create_tool_calling_agent`): plans tool usage.
5. **Executor** (`AgentExecutor`): runs the loop and returns output.

This is the base pattern to later expand into multi-agent systems.
