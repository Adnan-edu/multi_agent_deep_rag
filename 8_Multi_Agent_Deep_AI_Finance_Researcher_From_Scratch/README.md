# Multi-Agent Deep AI Finance Researcher from Scratch

## Overview

This project implements a sophisticated **Multi-Agent Deep AI Finance Researcher** designed to conduct deep financial analysis. It leverages a hierarchical multi-agent architecture where a central **Orchestrator Agent** manages a team of specialized **Researcher Agents** and an **Editor Agent** to deliver comprehensive financial reports.

The system is built to handle complex financial queries by breaking them down into thematic questions, conducting parallel research using hybrid retrieval strategies, and synthesizing the findings into a cohesive final output. It uniquely combines historical data analysis from SEC filings with live market data from Yahoo Finance.

## System Architecture

The core of the system is the hierarchical orchestration of agents. The **Orchestrator** plans the research, delegates tasks to **Researchers**, and coordinates the **Editor** to compile the final report.

![System Architecture](arch_custom_ai_research_agent.png)

## The Expert Team

The system employs three distinct types of agents, each with a specific role:

1.  **Orchestrator Agent**: The project manager. It analyzes the user's query, creates a research plan with specific thematic questions, delegates these to Researcher Agents, and ultimately triggers the Editor Agent.
2.  **Researcher Agent**: The diligent analyst. It executes specific research tasks using RAG tools for historical data and the Yahoo Finance MCP for live data. It writes detailed findings to dedicated research files.
3.  **Editor Agent**: The synthesizer. It reads all individual research notes and the original plan to compile a final, well-structured financial report (`report.md`) for the user.

![Expert Research Team](expert_research_team.png)

## Research Flow

From the initial user question to the final report, the data flows through a structured pipeline:
1.  **Question Analysis**: The Orchestrator breaks down the query.
2.  **Parallel Research**: Multiple Researchers gather data simultaneously.
3.  **Synthesis**: The Editor combines all findings.
4.  **Final Report**: The user receives a comprehensive answer.

![From Question to Report](flow_question_report.png)

## Models and Technologies

This project utilizes state-of-the-art models and tools to ensure high-quality research and retrieval.

### Large Language Models (LLM)
*   **Primary Model**: `gpt-4o` (OpenAI). This model is used for all agent reasoning, planning, and content generation.

### Embeddings and Retrieval (RAG)
The system employs a **Hybrid Retrieval** strategy to ensure no critical financial detail is missed.
*   **Dense Embeddings**: `intfloat/e5-large-v2` (via HuggingFaceEmbeddings).
*   **Sparse Embeddings**: `Qdrant/bm25` (via FastEmbedSparse).
*   **Vector Database**: **Qdrant**.
*   **Reranker**: `BAAI/bge-reranker-base` (via HuggingFaceCrossEncoder) is used to refine search results for maximum relevance.

### Tools and Integrations
*   **LangGraph**: For managing state and agent orchestration.
*   **LangChain**: For tool creation and model interaction.
*   **Yahoo Finance MCP (Model Context Protocol)**: A custom tool server (`yahoo-finance-mcp-server`) connected via `uvx` to provide real-time stock prices, news, financial statements, and analyst recommendations.
*   **Think Tool**: A strategic reflection tool allowing agents to pause and evaluate their findings before proceeding.

## Installation and Usage

1.  **Environment Setup**:
    Ensure you have Python installed and activate your virtual environment.
    ```bash
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    Install the required packages as listed in the project's requirements (e.g., `langgraph`, `langchain`, `qdrant-client`).
    *Note: The Yahoo Finance MCP requires `uv` to be installed.*

3.  **Configuration**:
    Set up your `.env` file with necessary API keys:
    *   `OPENAI_API_KEY`
    *   `QDRANT_URL` & `QDRANT_API_KEY` (if using cloud)

4.  **Running the Researcher**:
    Execute the main notebook `10 Multi-Agent Deep AI Finance Researcher.ipynb` to initialize the agents and start a research session.
