# 📈 MCP for Finance - Stock Researcher Agent

A comprehensive financial research assistant powered by **Model Context Protocol (MCP)** and **LangChain**. This agent leverages the Yahoo Finance MCP server to perform real-time stock analysis, competitor research, and market trend tracking, presenting the data in a rich, interactive Streamlit application.

![Agent Overview](Agent_Overview.png)

## 🚀 Overview

The **Stock Researcher Agent** is designed to bridge the gap between large language models and real-time financial data. By using the **Yahoo Finance MCP Server**, the agent can autonomously query stock prices, news, financial statements, and analyst recommendations.

*   **Architecture**:
    *   **Frontend**: [Streamlit](https://streamlit.io/) for a polished, collaborative user interface.
    *   **Orchestration**: [LangChain](https://www.langchain.com/) for agentic workflow and tool management.
    *   **Intelligence**: `gpt-4o-mini` (via LangChain OpenAI) for reasoning and data synthesis.
    *   **Data Layer**: `yahoo-finance-mcp-server` accessed via `langchain-mcp-adapters`.

## 📱 Stock Researcher App

![Stock Researcher App](stock_researcher_app.png)

The application provides a "fancy" and intuitive interface for your financial queries. Key features include:

*   **Intelligent Analysis**: Enter natural language queries (e.g., "Analyze Apple vs. Microsoft").
*   **Structured Reports**: View financial metrics in clean, styled data tables.
*   **Dynamic Visualizations**: Interactive charts for stock price history and comparative performance.
*   **Automated Cleanup**: The agent manages its own resources, ensuring clean shutdowns of MCP servers.

---
*Created by Deep Agent.*
