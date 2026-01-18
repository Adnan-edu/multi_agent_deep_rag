# Multimodal Financial Research Agent

This project implements an intelligent financial research agent capable of performing deep market analysis by combining **Hybrid RAG (Retrieval-Augmented Generation)** on historical SEC filings with **Live Market Data** research via Yahoo Finance MCP.

The agent is designed to bridge the gap between static historical data (10-K, 10-Q) and dynamic real-time market information, providing comprehensive financial insights.

## System Architecture

The system orchestrates an intelligent agent that can route queries between historical document search and live financial tools. It utilizes a modular design where specific agents or tools handle distinct data sources.

![Multimodal Research Agent Architecture](multimodal_research_agent_arch.png)

### Key Capabilities
*   **Dual-Source Intelligence**: Seamlessly integrates long-term historical context (SEC filings) with immediate market pulse (Live Stock Data).
*   **Intelligent Routing**: The agent decides whether to use RAG for historical data or calls the Yahoo Finance MCP for real-time updates.
*   **Memory & Persistence**: Uses `LangGraph` with SQLite checkpointing to maintain conversation state and memory across interactions.

## Search Mechanism: Hybrid RAG

The project employs a robust **Hybrid Search** strategy to ensure high-precision retrieval from financial documents.

![Hybrid Search Strategy](hybrid_search.png)

This approach combines:
1.  **Dense Retrieval**: Captures semantic meaning using embeddings.
2.  **Sparse Retrieval**: Captures exact keyword matches (e.g., specific terminologies, ticker symbols) using BM25.
3.  **Reranking**: Refines the retrieved results to ensure the most relevant chunks are passed to the Context Window.
4.  **Metadata Filtering**: Automatically extracts filters (Company, Fiscal Year, Quarter, Doc Type) from user queries to narrow down the search space.

## Models & Embeddings

Based on the analysis of the codebase, the following models and configurations are used:

| Component | Model / Technology | Description |
| :--- | :--- | :--- |
| **LLM (Reasoning)** | `gpt-4o-mini` | Used for the main agent, query analysis, and synthesis of results. |
| **Dense Embeddings** | `intfloat/e5-large-v2` | High-quality text embeddings for semantic search in Qdrant. |
| **Sparse Embeddings** | `Qdrant/bm25` | Sparse vector representation for keyword-based retrieval. |
| **Reranker** | `BAAI/bge-reranker-base` | Cross-encoder model used to score and rerank retrieved documents for better relevance. |
| **Vector Database** | `Qdrant` | Stores and manages the hybrid vectors (Dense + Sparse). |

## Project Structure

*   **`08_Multimodal_Research_Agent.ipynb`**: The main notebook that initializes the agent, defines tools, and demonstrates the workflow.
*   **`scripts/rag_tools.py`**: Contains the logic for `hybrid_search`, metadata extraction, and Qdrant client setup.
*   **`scripts/yahoo_mcp.py`**: Integrates with the Yahoo Finance MCP server to fetch live stock data, news, and financials.
*   **`scripts/schema.py`**: Defines Pydantic models for structured metadata extraction (Company, DocType, FiscalQuarter).
*   **`scripts/prompts.py`**: Stores system prompts for different agent personas (Orchestrator, Researcher, Editor).

## Setup & Usage

1.  **Environment Setup**: Ensure valid API keys for OpenAI and Qdrant are set in your environment or `.env` file.
2.  **Dependencies**: Install required packages as listed in the notebook (e.g., `langchain`, `qdrant-client`, `fastembed-gpu`).
3.  **Running the Agent**:
    *   Open `08_Multimodal_Research_Agent.ipynb`.
    *   Run the cells to initialize the Vector Store and Tools.
    *   Invoke the agent with natural language queries like:
        > "What was Apple's revenue in 2023?" (Trigger: Hybrid Search)
        > "What is the current stock price of Apple?" (Trigger: Live Finance Tool)

## Credits
Based on the **Deep Agent** course structure for building advanced Multi-Agent RAG systems.
