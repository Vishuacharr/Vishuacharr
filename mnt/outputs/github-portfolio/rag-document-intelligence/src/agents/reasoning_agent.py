"""Reasoning agent — drafts and refines answers from retrieved context."""

from __future__ import annotations
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI assistant. Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite which document sections support your answer."""),
    ("human", """Context:
{context}

Conversation history:
{history}

Question: {query}

Answer:"""),
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an editor. Improve the draft answer based on the critique while maintaining factual accuracy."),
    ("human", "Draft:\n{draft}\n\nCritique:\n{critique}\n\nImproved answer:"),
])


class ReasoningAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.draft_chain = DRAFT_PROMPT | llm
        self.refine_chain = REFINE_PROMPT | llm

    def draft_answer(self, query: str, docs: List[Dict[str, Any]], history: str = "") -> str:
        context = "\n\n".join(
            f"[Source {i+1}] {doc['content']}" for i, doc in enumerate(docs)
        )
        result = self.draft_chain.invoke({"context": context, "query": query, "history": history})
        return result.content.strip()

    def synthesize(self, draft: str, critique: str) -> str:
        result = self.refine_chain.invoke({"draft": draft, "critique": critique})
        return result.content.strip()
