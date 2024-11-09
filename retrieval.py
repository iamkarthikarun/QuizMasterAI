"""Retrieval module for hybrid search combining semantic and keyword-based approaches."""

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_URL"),
    "port": 6333,
    "collection": {"bge": "HistoryBook"},
    "search_type": "similarity_score_threshold",
    "n_results": 7,
    "score_threshold": 0.1,
}

BM25_CONFIG = {
    "filepath": os.getenv("BM25_FILEPATH"),
    "content_column": "indexedText",
    "pickle_load": True,
    "n_results": 7,
    "min_score": 2.5,
}

def initialize_qdrant(url: str, port: int, embedding_func: Any, collection_name: str) -> Qdrant:
    """Initialize Qdrant client and return vector store instance."""
    client = QdrantClient(
        url=url, 
        port=port, 
        verify=False, 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    return Qdrant(
        client=client, 
        embeddings=embedding_func, 
        collection_name=collection_name
    )

def initialize_bm25_retriever(filepath: str, content_col: str, n_results: int, pickle_load: bool) -> BM25Retriever:
    """Initialize BM25 retriever with custom top-n function."""
    def custom_get_top_n(self, query: str, documents: List[str], n: int = 5) -> Tuple[List[str], List[float]]:
        assert self.corpus_size == len(documents)
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        scores.sort()
        return [documents[i] for i in top_n], scores[::-1][:n]

    if pickle_load:
        import pickle
        with open(filepath, "rb") as f:
            retriever = pickle.load(f)
    else:
        df = pd.read_csv(filepath)
        docs = DataFrameLoader(df, page_content_column=content_col).load()
        retriever = BM25Retriever.from_documents(docs)
    
    retriever.k = n_results
    retriever.vectorizer.get_top_n = custom_get_top_n.__get__(
        retriever.vectorizer, BM25Okapi
    )
    return retriever

def retrieve_bm25_results(retriever: BM25Retriever, query: str, k: int, min_score: float) -> Tuple[List[str], List[float]]:
    """Retrieve and filter BM25 results based on minimum score."""
    docs, scores = retriever.get_relevant_documents(query)
    filtered_docs, filtered_scores = [], []
    
    for doc, score in zip(docs, scores):
        if score >= min_score:
            filtered_docs.append(doc)
            filtered_scores.append(score)
        else:
            break
    
    return filtered_docs, filtered_scores

def initialize_embedding_model(model_name: str = "Alibaba-NLP/gte-large-en-v1.5", device: str = "cpu") -> HuggingFaceBgeEmbeddings:
    """Initialize BGE embedding model."""
    model_kwargs = {"device": device, "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    
    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

def initialize_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Dict:
    """Initialize cross-encoder reranker model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}

embedding_model = initialize_embedding_model(device="cpu")
reranker = initialize_reranker()

qdrant_store = initialize_qdrant(
    url=QDRANT_CONFIG["url"],
    port=QDRANT_CONFIG["port"],
    embedding_func=embedding_model,
    collection_name=QDRANT_CONFIG["collection"]["bge"],
)

bm25_retriever = initialize_bm25_retriever(
    filepath=BM25_CONFIG["filepath"],
    content_col=BM25_CONFIG["content_column"],
    n_results=BM25_CONFIG["n_results"],
    pickle_load=BM25_CONFIG["pickle_load"],
)

def process_query(query: str) -> Tuple[str, List[float]]:
    """Process raw query and generate embedding."""
    query_processed = query.lower().replace("?", "")
    query_vector = embedding_model.embed_query(query)
    return query_processed, query_vector

def semantic_search(query_vector: List[float]) -> List[str]:
    """Perform semantic search using Qdrant."""
    client = QdrantClient(
        url=QDRANT_CONFIG["url"],
        api_key=os.getenv("QDRANT_API_KEY"),
        verify=False
    )
    results = client.search(
        collection_name="HistoryBook",
        query_vector=query_vector,
        limit=10
    )
    return [result.payload['summary'] for result in results]

def search(query: str) -> List[str]:
    """Execute hybrid search with reranking."""
    query_processed, query_vector = process_query(query)
    semantic_docs = semantic_search(query_vector)
    keyword_docs, _ = retrieve_bm25_results(
        bm25_retriever,
        query_processed,
        k=BM25_CONFIG["n_results"],
        min_score=BM25_CONFIG["min_score"],
    )
    
    all_docs = list(set(semantic_docs + [doc.page_content for doc in keyword_docs]))
    cross_inputs = [[query, doc] for doc in all_docs]
    
    inputs = reranker['tokenizer'](
        cross_inputs,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=1024
    )
    
    with torch.no_grad():
        scores = reranker['model'](**inputs, return_dict=True).logits.view(-1,).float()
    
    ranked_results = sorted(
        [{'id': idx, 'score': scores[idx]} for idx in range(len(scores))],
        key=lambda x: x['score'],
        reverse=True
    )
    
    return [all_docs[result['id']] for result in ranked_results[:5]]
