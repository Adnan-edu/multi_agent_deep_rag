# Image Description Generation with Multi-Modal LLMs

This part of the project focuses on transforming visual data extracted from financial PDFs into detailed, searchable textual descriptions using state-of-the-art Multi-Modal Large Language Models (LLMs).

## Overview

Complex financial documents often contain charts, graphs, and tables that are difficult for traditional text-based RAG pipelines to index. This project bridges that gap by using vision-capable LLMs like **Qwen2.5-VL** to generate comprehensive textual insights for every significant image extracted from the reports.

## Key Features

- **Multi-Modal Visual Understanding**: Utilizes advanced vision models to "read" charts, graphs, and document pages.
- **Specialized Financial Prompts**: Employs domain-specific prompting to focus on metrics, data points, trends (growth/decline), and row/column extraction from tables.
- **Automated Batch Processing**: Uses recursive directory scanning (`rglob`) to identify and process all page-level images across different companies and documents.
- **RAG-Ready Outputs**: Saves descriptions as structured Markdown files, enabling seamless integration into vector databases for enhanced retrieval.

## Workflow

1.  **Environment Setup**: Loads API keys and initializes the HuggingFace Inference Client for **Qwen2.5-VL**.
2.  **Asset Discovery**: Scans the `rag-data/images` directory for extracted PNG files.
3.  **Transformation**:
    - Converts images to base64 for LLM consumption.
    - Sends image + financial context prompt to the vision model.
    - Receives detailed analysis focusing on numbers and factual trends.
4.  **Persistence**: Writes descriptions to a parallel `images_desc` directory, mirrored by company and document name.

## Directory Structure

```text
data/rag-data/
├── images/        # Source PNG images (extracted in previous stage)
└── images_desc/   # Generated Markdown descriptions
    └── {company}/
        └── {document}/
            └── page_X.md
```

## Dependencies

- `huggingface_hub`: For using the **Qwen2.5-VL** multi-modal model.
- `Pillow`: Image processing and base64 conversion.
- `python-dotenv`: Management of environment variables and API keys.

---
*Developed as part of the Multi-Agent Deep RAG project.*
