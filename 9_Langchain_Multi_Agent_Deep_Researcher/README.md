# Deep Agent - Langchain's Multi-Agent Deep Researcher

Advanced financial research system using LangChain's DeepAgent with context isolation and strategic delegation. This project demonstrates a powerful architecture for conducting deep financial analysis by combining autonomous sub-agents, hybrid retrieval (RAG), and live market data integration.

## 🚀 Overview

This system is designed to perform complex financial research tasks by orchestrating specialized sub-agents. It features:

-   **DeepAgent Framework**: Leverages LangChain's advanced agent architecture for robust task execution.
-   **Context Isolation**: Sub-agents operate with isolated contexts to focus exclusively on their assigned research tasks, collecting independent evidence.
-   **Strategic Delegation**: A main orchestrator agent breaks down complex queries and delegates them to specialist sub-agents.
-   **Hybrid RAG + Live Data**: Combines rigorous historical data analysis (SEC filings via Qdrant) with real-time market data (Yahoo Finance MCP).
-   **File-Based Memory**: Persists research request and artifacts (reports, todos) using a secure filesystem backend with sandbox enforcement (`virtual_mode=True`).

## 🏗️ Architecture

The system operates on a hierarchical multi-agent structure:

1.  **Orchestrator Agent**:
    -   acts as the main interface for the user.
    -   Uses `think_tool` to plan research steps.
    -   Delegates specific financial questions to the `financial-research-agent`.
    -   Synthesizes findings into a final report.

2.  **Financial Research Sub-Agent**:
    -   **Role**: Dedicated specialist for gathering financial data.
    -   **Context**: Isolated execution environment to prevent context pollution between different research threads.
    -   **Tools**: Equipped with `hybrid_search` and `live_finance_researcher`.

3.  **Data & Memory**:
    -   **Vector Database**: Qdrant for storing and retrieving SEC filing chunks.
    -   **Memory**: SQLite checkpointer (`langgraph.checkpoint.sqlite`) for conversation history.
    -   **File Backend**: `FilesystemBackend` for storing research outputs (reports, markdown files) in a sandboxed directory.

## 🛠️ Technical Specifications

### Models & Embeddings

The system utilizes state-of-the-art models for reasoning and retrieval:

-   **Large Language Model (LLM)**:
    -   **Model**: `gpt-4o` (OpenAI)
    -   **Role**: Core reasoning engine for the Orchestrator and Sub-agents. Used for query planning, tool selection, and report synthesis.
    -   *Configuration*: Defined in `scripts/llm.py`. Capable of structured output extraction.

-   **Embeddings (Dense)**:
    -   **Model**: `intfloat/e5-large-v2` (HuggingFace)
    -   **Role**: Semantic search for finding contextually relevant passages in SEC filings.
    -   *Implementation*: `HuggingFaceEmbeddings` with normalized embeddings.

-   **Embeddings (Sparse)**:
    -   **Model**: `Qdrant/bm25`
    -   **Role**: Keyword-based search (BM25) to ensure precise matching of specific terms (financial metrics, years, company names).
    -   *Implementation*: `FastEmbedSparse`.

-   **Reranker**:
    -   **Model**: `BAAI/bge-reranker-base`
    -   **Role**: Re-ranks retrieval results to improve relevance before passing them to the LLM (referenced in `scripts/rag_tools.py`).

### Tools & Integrations

-   **Hybrid Search (`hybrid_search`)**:
    -   Combines Dense and Sparse embeddings using **Qdrant**'s Hybrid Retrieval Mode.
    -   Automatically extracts metadata filters (Company, Doc Type, Year, Quarter) from natural language queries using the LLM.
    -   Targets `10-K` (Annual), `10-Q` (Quarterly), and `8-K` (Current) reports.

-   **Live Finance Researcher (`live_finance_researcher`)**:
    -   Connects to **Yahoo Finance** via Model Context Protocol (MCP).
    -   Provides real-time stock prices, news, analyst recommendations, and option chains.
    -   Executed safely via `subprocess` calling `scripts/yahoo_mcp.py`.

-   **Think Tool (`think_tool`)**:
    -   Enables agents to record a "thought" step, forcing a pause for reflection on current findings before proceeding.

## 📂 Project Structure

```
├── 11_Deep_Agent_Multi_Agent_Deep_Finance_Researcher.ipynb  # Main notebook to run the agent
├── scripts/
│   ├── agent_utils.py    # Utilities for streaming agent responses
│   ├── deep_prompts.py   # Detailed system instructions and prompts
│   ├── llm.py            # LLM initialization (OpenAI/Gemini config)
│   ├── rag_tools.py      # RAG tools: hybrid_search, live_finance_researcher
│   ├── schema.py         # Pydantic schemas for metadata extraction
│   └── yahoo_mcp.py      # Yahoo Finance MCP server client
└── research_outputs/     # Directory where agent saves reports (auto-generated)
```

## 📦 Installation

1.  **Environment Setup**:
    Ensure you have Python 3.10+ installed.

2.  **Install Dependencies**:
    ```bash
    pip install -U deepagents langchain langchain-community sentence-transformers qdrant-client langchain-qdrant fastembed fastembed-gpu langgraph-checkpoint-sqlite langchain-openai langchain-mcp-adapters langchain-google-genai
    ```

3.  **Environment Variables**:
    Create a `.env` file or set the following variables:
    ```bash
    OPENAI_API_KEY=your_openai_key
    GOOGLE_API_KEY=your_google_key (optional, if using Gemini)
    QDRANT_URL=http://localhost:6333 (or your Qdrant Cloud URL)
    QDRANT_API_KEY=your_qdrant_key (if using Qdrant Cloud)
    ```

## 🚦 Usage

1.  **Running the Agent**:
    Open `11_Deep_Agent_Multi_Agent_Deep_Finance_Researcher.ipynb` and run the cells to initialize the DeepAgent.

2.  **Example Query**:
    ```python
    from scripts.agent_utils import stream_agent_response

    # Define your research question
    query = "Compare Apple and Amazon's 2024 revenue and profitability."
    user_id = "user_ss1"
    thread_id = "session_01"

    # Get the agent instance
    agent = get_deep_agent(user_id, thread_id)

    # Run the stream
    stream_agent_response(agent, query, thread_id)
    ```

3.  **Output**:
    The agent will:
    *   Create a plan (todos).
    *   Delegate to the research sub-agent.
    *   Perform hybrid searches and live data lookups.
    *   Synthesize a final report in markdown format saved to `research_outputs/{user_id}/{thread_id}/final_report.md`.
