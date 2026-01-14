# Multimodal Data Extraction with Docling

This project focuses on extracting high-quality, structured data from complex financial PDF documents using the **Docling** library. It automates the conversion of unstructured PDFs into machine-readable formats suitable for RAG (Retrieval-Augmented Generation) pipelines.

## Project Overview

The objective is to process financial reports (like 10-K, 10-Q) and extract not just text, but also visual and tabular data while maintaining context and structural integrity.

## Key Features

- **Automated Metadata Extraction**: Parses filenames (e.g., `Apple 10-K 2023.pdf`) to identify the company, document type, and fiscal period.
- **Markdown Conversion**: Converts PDF content to Markdown with explicit page breaks (`<!-- page break -->`) for precise chunking.
- **Intelligent Image Extraction**: Identifies and saves significant visual elements (charts, diagrams) based on size thresholds (e.g., >500x500 pixels).
- **Context-Aware Table Extraction**: Extracts tables into individual Markdown files, preserving the preceding context (paragraphs) and page metadata for better searchability.

## Workflow

1.  **Setup**: Configure directory paths for input PDFs and categorized outputs.
2.  **Conversion**: Initialize `DocumentConverter` with specialized `PdfPipelineOptions`.
3.  **Processing**:
    - Iterate through document items to locate pictures and tables.
    - Filter and save high-resolution images.
    - Segment and save tables with a 2-paragraph "look-back" context.
4.  **Export**: Write the full document Markdown and granular table/image assets to structured subdirectories.

## Directory Structure

Processed data is organized logically by company and document:

```text
data/rag-data/
├── markdown/  # Full document text
├── images/    # High-res charts and diagrams
└── tables/    # Extracted tables with context
```

## Requirements

- `docling`: Core library for document conversion.
- `docling-core`: Schema and type definitions.
- `pathlib`: For robust filesystem interactions.

---
*Developed as part of the Multi-Agent Deep RAG project.*
