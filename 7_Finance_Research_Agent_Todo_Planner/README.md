# Deep Finance Researcher with TODO Planner

## Overview
This project implements an intelligent **Deep Finance Research Agent** equipped with a **TODO Planner** and **Summarization** capabilities. The agent is designed to perform comprehensive financial research by combining **Historical Data RAG** (Retrieval-Augmented Generation) from SEC filings with **Live Market Data** from Yahoo Finance.

The system leverages a multi-agent architecture (orchestrated via middleware) to manage tasks, summarize long conversations, and persistently store state, allowing for complex, multi-step financial analysis.

## Agent Architecture
The agent is built using **LangChain** and **LangGraph**, utilizing specialized middleware for task management and memory optimization.

![Agent with Todo Planner](Agent_With_Todo_Planner.png)

### Key Features
*   **Hybrid Search RAG**: Retrieves information from historical financial documents (10-Ks, 10-Qs) using semantic and keyword search.
*   **Live Market Research**: Accesses real-time stock prices, news, and analyst recommendations via Yahoo Finance.
*   **TODO Planner**: Uses `TodoListMiddleware` to break down complex user queries into actionable sub-tasks and track progress.
*   **Memory Management**: Implements `SummarizationMiddleware` to condense conversation history, ensuring the context remains relevant without exceeding token limits.
*   **Persistence**: Uses `SqliteSaver` to checkpoint agent state, allowing sessions to be paused and resumed.

## Models & Embeddings
The system utilizes state-of-the-art models for logic, embeddings, and retrieval:

*   **Large Language Model (LLM)**:
    *   **GPT-4o-mini**: Used as the core brain for the agent, driving the Orchestrator, Researcher, and Editor roles, as well as handling tool selection and response generation.
    
*   **Embeddings**:
    *   **Dense Embeddings**: `intfloat/e5-large-v2` (via Hugging Face) for semantic understanding of financial documents.
    *   **Sparse Embeddings**: `Qdrant/bm25` (via FastEmbed) for keyword-based retrieval.

*   **Reranking** (Configuration):
    *   `BAAI/bge-reranker-base` is defined in the configuration for optimizing retrieval results (Cross-Encoder Re-ranking).

## Tools & Technologies

### Core Components
*   **Vector Database**: **Qdrant** is used to store and retrieve dense and sparse embeddings, enabling efficient Hybrid Search.
*   **Orchestration**: **LangChain** & **LangGraph** for managing agent workflows and state.
*   **MCP Integration**: **Yahoo Finance MCP Server** integration via `MultiServerMCPClient` for fetching live financial data.

### Project Structure
*   **`scripts/`**: Contains the core logic modules:
    *   `agent_utils.py`: Utilities for streaming agent responses.
    *   `prompts.py`: Defines system prompts for the Orchestrator, Researcher, and Editor agents.
    *   `rag_tools.py`: Implements the `hybrid_search` tool and vectors store connection (Qdrant).
    *   `yahoo_mcp.py`: Wraps the Yahoo Finance MCP server for live data fetching.
    *   `schema.py`: Pydantic models for data validation and structure.
*   **Notebook**: `09_Deep_Finance_Researcher_with_TODO_Planner.ipynb` serves as the main entry point for initializing, configuring, and testing the agent.

## Usage
The agent operates by receiving natural language queries. It determines whether to fetch historical data (via RAG) or live data (via Yahoo Finance), plans its steps using the TODO middleware, and synthesizes a final response.

```python
# Example Usage
stream_agent_response(agent, "What is Amazon's revenue in Q1 2024?", thread_id="session_1")
```
