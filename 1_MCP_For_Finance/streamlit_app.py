import streamlit as st
import asyncio
import json
import pandas as pd
import sys
import os

# Ensure scripts directory is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from yahoo_mcp import finance_research

st.set_page_config(layout="wide", page_title="Stock Researcher Agent", page_icon="📈")

# Custom CSS for fancy tables and UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
    }
    .stButton button:hover {
        background-color: #155a8a;
        color: white;
    }
    
    /* Fancy Table Styling */
    .dataframe {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        text-align: left;
        padding: 12px 15px;
    }
    .dataframe td {
        background-color: #ffffff;
        color: #333;
        padding: 10px 15px;
        border-bottom: 1px solid #eeeeee;
    }
    .dataframe tr:nth-child(even) td {
        background-color: #f8f9fa;
    }
    .dataframe tr:hover td {
        background-color: #e2e6ea;
        transition: background-color 0.2s;
    }
    
    /* Metrics Box */
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Researcher Agent")
st.markdown("Analyze stocks, competitors, and market trends with an AI agent powered by Yahoo Finance MCP.")

# User Input
default_query = "Analyze Apple (AAPL) stock and its competitors like MSFT and Google. Present data clearly in the table."
query = st.text_area("Enter your research query:", value=default_query, height=100)

if st.button("Run Analysis"):
    with st.spinner("Agent is researching... This may take a moment."):
        try:
            # Instructions for JSON format
            instructions = """
            Provide the output STRICTLY in valid JSON format. Do not wrap in markdown code blocks (like ```json).
            The JSON structure should be:
            {
                "summary": "Markdown text summary of the analysis...",
                "tables": [
                    {
                        "title": "Table Title",
                        "columns": ["Col1", "Col2", ...],
                        "data": [
                            {"Col1": "Val1", "Col2": "Val2", ...},
                            ...
                        ]
                    }
                ],
                "charts": [
                    {
                        "title": "Chart Title",
                        "type": "line",
                        "data": [
                            {"Date": "2023-01-01", "Symbol": "AAPL", "Price": 150.0},
                            ...
                        ]
                    }
                ]
            }
            Ensure all numeric data is properly properly typed (numbers not strings) where possible, especially for charts.
            For chart data, ensure 'Date' is present if it's a time series.
            """
            
            # Run the async agent function
            response_text = asyncio.run(finance_research(query, instructions=instructions))
            
            # Clean response if it has markdown blocks
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                data = json.loads(response_text)
                
                # 1. Summary
                st.subheader("📝 Analysis Summary")
                st.markdown(data.get("summary", "No summary provided."))
                
                # 2. Tables
                tables = data.get("tables", [])
                if tables:
                    st.subheader("📊 Fundamental Data")
                    for table in tables:
                        st.markdown(f"**{table.get('title', 'Data Table')}**")
                        df = pd.DataFrame(table.get("data", []))
                        if not df.empty:
                            if "columns" in table:
                                # Ensure column order matches
                                df = df[table["columns"]]
                            # Use custom HTML for styling instead of standard st.dataframe for "fancy" look
                            st.markdown(df.to_html(classes="dataframe", index=False), unsafe_allow_html=True)
                        else:
                            st.info("No data in table.")
                        st.write("") # Spacer

                # 3. Charts
                charts = data.get("charts", [])
                if charts:
                    st.subheader("📈 Vizualizations")
                    for chart in charts:
                        st.markdown(f"**{chart.get('title', 'Chart')}**")
                        chart_data = chart.get("data", [])
                        if chart_data:
                            df_chart = pd.DataFrame(chart_data)
                            # Attempt to parse date
                            if "Date" in df_chart.columns:
                                df_chart["Date"] = pd.to_datetime(df_chart["Date"])
                            
                            # Pivot for multi-line chart if multiple symbols
                            if "Symbol" in df_chart.columns and "Price" in df_chart.columns:
                                df_pivot = df_chart.pivot(index="Date", columns="Symbol", values="Price")
                                st.line_chart(df_pivot)
                            elif "Close" in df_chart.columns: # Single ticker generic
                                st.line_chart(df_chart.set_index("Date")["Close"])
                            else:
                                st.bar_chart(df_chart)
                        else:
                            st.info("No data for chart.")

            except json.JSONDecodeError:
                st.error("Failed to parse agent response as JSON. Showing raw output:")
                st.markdown(response_text)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(e)
