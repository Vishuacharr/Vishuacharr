"""Retrieval agent — rewrites query and fetches relevant chunks from pgvector."""

from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a query optimization expert. Rewrite the user query to maximize retrieval recall."),
    ("human", "Original query: {query}\n\nRewritten query:"),
])


class RetrievalAgent:
    def __init__(self, vectorstore, llm: ChatOpenAI, rewrite: bool = True):
        self.vectorstore = vectorstore
        self.llm = llm
        self.rewrite = rewrite
        self.chain = REWRITE_PROMPT | llm

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        search_query = query
        if self.rewrite:
            rewritten = self.chain.invoke({"query": query})
            search_query = rewritten.content.strip()

        docs = self.vectorstore.similarity_search_with_score(search_query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in docs
        ]
