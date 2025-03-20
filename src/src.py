import os
import json
import bs4
import numpy as np
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain.load import dumps, loads

def load_config(config_path="config.json"):
    """ Load configuration from a JSON file """
    with open(config_path, "r") as config_file:
        return json.load(config_file)

def setup_environment(config):
    """ Set up API keys """
    os.environ['LANGCHAIN_API_KEY'] = config.get("LANGCHAIN_API_KEY", "")
    os.environ['OPENAI_API_KEY'] = config.get("OPENAI_API_KEY", "")

def load_documents(config):
    """ Load documents from a web source """
    loader = WebBaseLoader(web_paths=[config["DOCUMENT_SOURCE"]])
    return loader.load()

def process_documents(docs, config):
    """ Split documents into smaller chunks """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["CHUNK_SIZE"], chunk_overlap=config["CHUNK_OVERLAP"])
    return text_splitter.split_documents(docs)

def create_vectorstore(splits):
    """ Create vector store and retriever """
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

def multi_query_retrieval(retriever):
    """ Perform Multi-Query Retrieval """
    query_template = """You are an AI assistant. Generate five different versions of the question:\n\nOriginal question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(query_template)
    
    generate_queries = (
        prompt_perspectives | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
    )
    
    def get_unique_union(documents):
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        unique_docs = list(set(flattened_docs))
        return [loads(doc) for doc in unique_docs]
    
    return generate_queries | retriever.map() | get_unique_union

def reciprocal_rank_fusion(results, k=60):
    """ Apply Reciprocal Rank Fusion """
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    
    return [(loads(doc), score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]

def final_rag_chain(retrieval_chain):
    """ Define the final RAG chain """
    final_template = """Answer the following question based on this context:\n\n{context}\n\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(final_template)
    
    llm = ChatOpenAI(temperature=0)
    return (
        {"context": retrieval_chain, "question": itemgetter("question")} 
        | prompt
        | llm
        | StrOutputParser()
    )

def main():
    config = load_config()
    setup_environment(config)
    docs = load_documents(config)
    splits = process_documents(docs, config)
    retriever = create_vectorstore(splits)
    retrieval_chain = multi_query_retrieval(retriever)
    
    question = "What is task decomposition for LLM agents?"
    docs_retrieved = retrieval_chain.invoke({"question": question})
    print(f"Retrieved {len(docs_retrieved)} documents")
    
    retrieval_chain_rag_fusion = retrieval_chain | reciprocal_rank_fusion
    docs_fusion = retrieval_chain_rag_fusion.invoke({"question": question})
    print(f"RAG Fusion retrieved {len(docs_fusion)} documents")
    
    final_chain = final_rag_chain(retrieval_chain_rag_fusion)
    response = final_chain.invoke({"question": question})
    print(response)

if __name__ == "__main__":
    main()
