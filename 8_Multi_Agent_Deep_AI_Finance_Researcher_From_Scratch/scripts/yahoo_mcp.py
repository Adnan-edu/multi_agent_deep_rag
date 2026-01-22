"""Yahoo Finance MCP module with LangChain integration.

This module connects to the Yahoo Finance MCP server via uvx
and provides finance research capabilities using Yahoo Finance tools.

Usage:
    python -m scripts.yahoo_mcp

    or

    from scripts.yahoo_mcp import finance_research, get_tools
"""

######## MCP SETUP ###############
# MCP GITHUB
# https://github.com/laxmimerit/MCP-Mastery-with-Claude-and-Langchain
# https://github.com/laxmimerit/Agentic-RAG-with-LangGraph-and-Ollama

# https://github.com/langchain-ai/langchain-mcp-adapters
# https://github.com/laxmimerit/yahoo-finance-mcp-server

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import gc
import signal
import atexit
import subprocess
from typing import Optional

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

import asyncio
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from scripts.llm import llm


# System prompt for the Yahoo Finance research agent
YAHOO_SYSTEM_PROMPT = """
You are a financial research assistant helping users analyze stocks and financial data using Yahoo Finance.

Available Tools:
- get_historical_stock_prices: Get historical stock prices (ticker required, optional: period='1mo', interval='1d')
- get_stock_info: Get comprehensive stock information including price, metrics, financials, etc.
- get_yahoo_finance_news: Get latest news for a stock ticker
- get_stock_actions: Get dividends and stock splits information
- get_financial_statement: Get financial statements (ticker and financial_type required: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow)
- get_holder_info: Get holder information (ticker and holder_type required: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders)
- get_option_expiration_dates: Get available option expiration dates
- get_option_chain: Get option chain data (ticker, expiration_date, option_type required: 'calls' or 'puts')
- get_recommendations: Get analyst recommendations (ticker and recommendation_type required: recommendations, upgrades_downgrades, optional: months_back=12)

Instructions:
- ALWAYS start by calling relevant tools to gather financial data when user asks about stocks
- Extract ticker symbol from user query (e.g., AAPL, MSFT, GOOGL)
- For general stock inquiries, start with get_stock_info to get comprehensive data
- For price analysis, use get_historical_stock_prices with appropriate period
- For news and sentiment, use get_yahoo_finance_news
- Present data in a clear, organized format with key insights highlighted
- Include specific numbers, percentages, and trends in your analysis
- Be proactive - gather data first, then provide comprehensive analysis
"""

# Global client reference for cleanup
_mcp_client: Optional[MultiServerMCPClient] = None
_uvx_processes: list = []


# ==================== CLEANUP FUNCTIONS ====================

def kill_uvx_processes():
    """Kill any running uvx yahoo-finance-mcp-server processes."""
    try:
        # Find and kill uvx processes related to yahoo-finance-mcp-server
        result = subprocess.run(
            ["pgrep", "-f", "yahoo-finance-mcp-server"],
            capture_output=True,
            text=True
        )
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"[Cleanup] Terminated yahoo-finance-mcp-server process {pid}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        print(f"[Cleanup] Error killing uvx processes: {e}")


def clear_uv_cache():
    """Clear uv/uvx cache to free up disk space."""
    try:
        cache_dir = os.path.expanduser("~/.cache/uv")
        if os.path.exists(cache_dir):
            # Only clear if explicitly requested (uncomment if needed)
            # subprocess.run(["rm", "-rf", cache_dir], check=True)
            # print(f"[Cleanup] Cleared uv cache at {cache_dir}")
            print(f"[Cleanup] UV cache exists at {cache_dir} (not cleared)")
    except Exception as e:
        print(f"[Cleanup] Error clearing uv cache: {e}")


def cleanup():
    """Comprehensive cleanup function for the Yahoo Finance MCP client."""
    global _mcp_client, _uvx_processes
    
    print("\n[Cleanup] Starting cleanup process...")
    
    # 1. Close MCP client
    _mcp_client = None
    print("[Cleanup] MCP client reference cleared")
    
    # 2. Kill any tracked uvx processes
    for proc in _uvx_processes:
        try:
            if proc.poll() is None:  # Process is still running
                proc.terminate()
                proc.wait(timeout=5)
                print(f"[Cleanup] Terminated tracked process {proc.pid}")
        except Exception as e:
            print(f"[Cleanup] Error terminating process: {e}")
    _uvx_processes.clear()
    
    # 3. Kill any remaining uvx yahoo-finance processes
    kill_uvx_processes()
    
    # 4. Run garbage collection
    gc.collect()
    print("[Cleanup] Garbage collection completed")
    
    # 5. Optional: Clear uv cache (disabled by default)
    # clear_uv_cache()
    
    print("[Cleanup] Cleanup completed successfully!")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    sig_name = signal.Signals(signum).name
    print(f"\n[Signal] Received {sig_name}, shutting down...")
    cleanup()
    sys.exit(0)


# Register cleanup functions
atexit.register(cleanup)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ==================== MCP FUNCTIONS ====================

async def get_tools():
    """
    Connect to the Yahoo Finance MCP server and retrieve available tools.
    
    Returns:
        list: List of LangChain-compatible tools from the Yahoo Finance MCP server.
    """
    global _mcp_client
    
    try:
        _mcp_client = MultiServerMCPClient(
            {
                "yahoo-finance": {
                    "command": "uvx",
                    "args": ["yahoo-finance-mcp-server"],
                    "transport": "stdio",
                }
            }
        )

        tools = await _mcp_client.get_tools()

        if not tools:
            print("[Warning] No tools loaded from Yahoo Finance MCP server")
        
        return tools
        
    except Exception as e:
        print(f"[Error] Failed to connect to Yahoo Finance MCP server: {e}")
        raise


async def finance_research(query: str, verbose: bool = True) -> str:
    """
    Perform financial research using the Yahoo Finance MCP tools.
    
    Args:
        query: The financial research question or request.
        verbose: Whether to print the response to console.
        
    Returns:
        str: The research response from the agent.
    """
    try:
        tools = await get_tools()
        
        if not tools:
            return "[Error] No tools available from Yahoo Finance MCP server."

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=YAHOO_SYSTEM_PROMPT
        )

        result = await agent.ainvoke({"messages": [HumanMessage(query)]})

        response = result["messages"][-1].text

        if verbose:
            print(response)

        return response
        
    except Exception as e:
        error_msg = f"[Error] Finance research failed: {e}"
        if verbose:
            print(error_msg)
        return error_msg


# ==================== EXAMPLE QUERIES ====================

EXAMPLE_QUERIES = [
    "What is the current stock price and recent performance of Apple (AAPL)? Also show me the latest news.",
    "Show me the historical stock prices for Microsoft (MSFT) over the past 3 months",
    "Get the latest news about Tesla (TSLA)",
    "What are the analyst recommendations for NVIDIA (NVDA)?",
    "Show me Apple's quarterly income statement",
    "Who are the major holders of Amazon (AMZN)?",
]


async def run_demo():
    """Run a demo query to test the Yahoo Finance MCP integration."""
    print("=" * 60)
    print("Yahoo Finance MCP Research Demo")
    print("=" * 60)
    
    query = EXAMPLE_QUERIES[0]
    print(f"\n[Query] {query}\n")
    print("-" * 60)
    
    try:
        await finance_research(query)
    except Exception as e:
        print(f"[Error] Demo failed: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    # Run the demo
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received")
    finally:
        cleanup()
