# Multi-Query RAG and RAG Fusion

## Overview
This repository provides a Python script to run Multi-Query RAG (Retrieval-Augmented Generation) and RAG Fusion using LangChain and OpenAI models. It improves document retrieval and ranking by:
- Generating multiple query variations to enhance retrieval effectiveness.
- Using Reciprocal Rank Fusion (RRF) to merge results from multiple queries, improving overall relevance.

## Features
- **Multi-Query Retrieval**: Expands a single query into multiple variations to retrieve a more diverse set of relevant documents.
- **RAG Fusion**: Uses Reciprocal Rank Fusion (RRF) to merge retrieved results, boosting relevance and recall.
- **End-to-End Pipeline**: Covers document indexing, retrieval, fusion, and final response generation.

## Setup
### Prerequisites
Ensure you have Python installed, then install the required dependencies:

```sh
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
```

### API Keys
Set up your API keys for OpenAI and LangChain in the environment variables:

```sh
export LANGCHAIN_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
```

Alternatively, set them in the script before running.

## Usage
### Running the Script
Run the Python script directly:

```sh
python src.py
```

This will:
1. Load and split documents.
2. Index them using ChromaDB.
3. Perform Multi-Query Retrieval.
4. Apply RAG Fusion for enhanced ranking.
5. Generate final responses using GPT-3.5/4.

## Flow
1. **Indexing**: Extracts and splits documents, embedding them into a vector database.
2. **Multi-Query Retrieval**: Reformulates the query into multiple variations to improve recall.
3. **RAG Fusion**: Merges and re-ranks retrieved results.
4. **Response Generation**: Uses an LLM to generate an answer based on the retrieved documents.

## References
- [LangChain Multi-Query Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever)
- [LangChain RAG Fusion](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb)
- [Towards Data Science - RAG Fusion](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)

## License
MIT License
