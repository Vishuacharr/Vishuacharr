"""
Multi-agent RAG orchestrator using LangGraph.
Coordinates 5 specialized agents: Retrieval, Reasoning, Critique, Synthesis, Memory.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .critique_agent import CritiqueAgent
from .memory import AgentMemory


# ---------------------------------------------------------------------------
# Shared graph state
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    query: str
    retrieved_docs: List[Dict[str, Any]]
    draft_answer: str
    critique: str
    final_answer: str
    is_faithful: bool
    iteration: int


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class RAGOrchestrator:
    """
    LangGraph-based multi-agent orchestration.

    Graph flow:
        retrieve → reason → critique → [synthesize | retrieve (retry)]
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        vectorstore,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=True)
        self.memory = AgentMemory()

        self.retrieval_agent = RetrievalAgent(vectorstore=vectorstore, llm=self.llm)
        self.reasoning_agent = ReasoningAgent(llm=self.llm)
        self.critique_agent = CritiqueAgent(llm=self.llm)

        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        builder = StateGraph(RAGState)

        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("reason", self._reason_node)
        builder.add_node("critique", self._critique_node)
        builder.add_node("synthesize", self._synthesize_node)

        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "reason")
        builder.add_edge("reason", "critique")
        builder.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"retry": "retrieve", "synthesize": "synthesize"},
        )
        builder.add_edge("synthesize", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _retrieve_node(self, state: RAGState) -> Dict:
        docs = self.retrieval_agent.retrieve(state["query"], k=5)
        return {"retrieved_docs": docs}

    def _reason_node(self, state: RAGState) -> Dict:
        draft = self.reasoning_agent.draft_answer(
            query=state["query"],
            docs=state["retrieved_docs"],
            history=self.memory.get_history(),
        )
        return {"draft_answer": draft}

    def _critique_node(self, state: RAGState) -> Dict:
        critique, is_faithful = self.critique_agent.evaluate(
            query=state["query"],
            answer=state["draft_answer"],
            docs=state["retrieved_docs"],
        )
        return {
            "critique": critique,
            "is_faithful": is_faithful,
            "iteration": state.get("iteration", 0) + 1,
        }

    def _synthesize_node(self, state: RAGState) -> Dict:
        final = self.reasoning_agent.synthesize(
            draft=state["draft_answer"],
            critique=state["critique"],
        )
        self.memory.add(role="user", content=state["query"])
        self.memory.add(role="assistant", content=final)
        return {"final_answer": final}

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route_after_critique(self, state: RAGState) -> str:
        if state["is_faithful"] or state.get("iteration", 0) >= self.MAX_ITERATIONS:
            return "synthesize"
        return "retry"

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        initial_state: RAGState = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "retrieved_docs": [],
            "draft_answer": "",
            "critique": "",
            "final_answer": "",
            "is_faithful": False,
            "iteration": 0,
        }
        result = self.graph.invoke(initial_state)
        return result["final_answer"]
