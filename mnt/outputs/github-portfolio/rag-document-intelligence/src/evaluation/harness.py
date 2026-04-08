"""
Evaluation harness using RAGAS metrics:
  - Faithfulness
  - Answer Relevance
  - Context Precision
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision


class EvaluationHarness:
    """Run RAGAS evaluation against the RAG orchestrator."""

    def run(
        self,
        questions: List[str],
        orchestrator: Any,
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        answers, contexts = [], []

        for q in questions:
            answer = orchestrator.query(q)
            retrieved = orchestrator.retrieval_agent.retrieve(q, k=5)
            answers.append(answer)
            contexts.append([doc["content"] for doc in retrieved])

        dataset_dict: Dict[str, Any] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            dataset_dict["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(dataset_dict)
        metrics = [faithfulness, answer_relevancy]
        if ground_truths:
            metrics.append(context_precision)

        result = evaluate(dataset, metrics=metrics)

        return {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
            "context_precision": round(float(result.get("context_precision", 0)), 4),
            "num_questions": len(questions),
        }
