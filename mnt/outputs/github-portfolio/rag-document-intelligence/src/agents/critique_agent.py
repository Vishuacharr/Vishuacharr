"""Critique agent — evaluates answer faithfulness against retrieved context."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checker. Evaluate if the answer is fully supported by the context.
Respond with JSON: {{"is_faithful": true/false, "issues": ["list of issues if any"], "score": 0.0-1.0}}"""),
    ("human", """Context:
{context}

Question: {query}
Answer: {answer}

Evaluation (JSON):"""),
])


class CritiqueAgent:
    def __init__(self, llm: ChatOpenAI, faithfulness_threshold: float = 0.80):
        self.llm = llm
        self.threshold = faithfulness_threshold
        self.chain = CRITIQUE_PROMPT | llm

    def evaluate(
        self,
        query: str,
        answer: str,
        docs: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        context = "\n\n".join(doc["content"] for doc in docs)
        result = self.chain.invoke({"context": context, "query": query, "answer": answer})

        try:
            data = json.loads(result.content.strip())
            is_faithful = data.get("is_faithful", False) or data.get("score", 0) >= self.threshold
            critique = "; ".join(data.get("issues", [])) or "Answer is faithful to context."
        except (json.JSONDecodeError, AttributeError):
            is_faithful = True
            critique = "Critique parsing failed — treating as faithful."

        return critique, is_faithful
