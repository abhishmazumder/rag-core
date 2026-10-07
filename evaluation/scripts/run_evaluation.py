"""Run retrieval evaluation: POST /query for each frozen evaluation query, then score it.

Reads evaluation_dataset.json from a run directory (never modifies it) and writes
evaluation_report.json next to it. External HTTP client of the API; imports no rag_core.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))

import retrieval_metrics as m  # noqa: E402

DATASET_FILE = "evaluation_dataset.json"
REPORT_FILE = "evaluation_report.json"
REPORT_VERSION = "v1"
METHOD = "vector"
KS = (1, 2, 3, 4)
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120.0


class EvaluationRunError(Exception):
    pass


class DatasetQuery(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    answerable: bool
    relevance: dict[str, int]


class EvaluationDataset(BaseModel):
    version: str
    run_id: str
    queries: list[DatasetQuery] = Field(min_length=1)


class QueryResponseChunk(BaseModel):
    id: str = Field(min_length=1)


class QueryResponse(BaseModel):
    chunks: list[QueryResponseChunk]


def load_dataset(path: Path) -> EvaluationDataset:
    try:
        return EvaluationDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except OSError as error:
        raise EvaluationRunError(f"Cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise EvaluationRunError(f"{path} is not valid JSON: {error}") from error
    except ValidationError as error:
        raise EvaluationRunError(f"Invalid evaluation dataset {path}:\n{error}") from error


def retrieve_chunk_ids(client: httpx.Client, query: DatasetQuery, top_k: int) -> list[str]:
    body = {
        "query": query.query,
        "candidate_id": query.candidate_id,
        "method": METHOD,
        "top_k": top_k,
    }
    try:
        response = client.post("/query", json=body)
        response.raise_for_status()
        parsed = QueryResponse.model_validate(response.json())
    except httpx.HTTPError as error:
        raise EvaluationRunError(f"POST /query failed for {query.id}: {error}") from error
    except (ValueError, ValidationError) as error:
        raise EvaluationRunError(f"Invalid POST /query response for {query.id}: {error}") from error
    return [chunk.id for chunk in parsed.chunks]


def score_query(query: DatasetQuery, retrieved_ids: list[str]) -> dict[str, object]:
    unknown = [cid for cid in retrieved_ids if cid not in query.relevance]
    if unknown:
        raise EvaluationRunError(
            f"Query {query.id}: API returned chunk IDs not in the evaluation dataset: {unknown}"
        )
    scores = [query.relevance[cid] for cid in retrieved_ids]
    ground_truth = list(query.relevance.values())
    total_relevant = sum(1 for score in ground_truth if score > 0)

    result: dict[str, object] = {
        "query_id": query.id,
        "candidate_id": query.candidate_id,
        "method": METHOD,
        "retrieved_chunk_ids": retrieved_ids,
        "relevance_scores_in_retrieval_order": scores,
    }
    for k in KS:
        result[f"precision_at_{k}"] = m.precision_at_k(scores, k)
    for k in KS:
        result[f"recall_at_{k}"] = m.recall_at_k(scores, total_relevant, k)
    result["reciprocal_rank"] = m.reciprocal_rank(scores)
    for k in KS:
        result[f"ndcg_at_{k}"] = m.ndcg_at_k(scores, ground_truth, k)
    return result


def metric_names() -> list[str]:
    return (
        [f"precision_at_{k}" for k in KS]
        + [f"recall_at_{k}" for k in KS]
        + ["reciprocal_rank"]
        + [f"ndcg_at_{k}" for k in KS]
    )


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    means: dict[str, float | None] = {}
    defined_counts: dict[str, int] = {}
    for name in metric_names():
        values = [r[name] for r in results]
        means[name] = m.aggregate_mean(values)
        defined_counts[name] = sum(1 for v in values if v is not None)
    return {"mean": means, "queries_with_defined_value": defined_counts}


def run_evaluation(
    run_dir: Path,
    client: httpx.Client,
    base_url: str,
    top_k: int,
    overwrite: bool = False,
) -> Path:
    if top_k < max(KS):
        raise EvaluationRunError(f"--top-k must be >= {max(KS)}")
    dataset_path = run_dir / DATASET_FILE
    report_path = run_dir / REPORT_FILE
    if not dataset_path.is_file():
        raise EvaluationRunError(f"{DATASET_FILE} not found in {run_dir}")
    if report_path.exists() and not overwrite:
        raise EvaluationRunError(f"{report_path} already exists; use --overwrite to replace it")

    dataset = load_dataset(dataset_path)
    results = [
        score_query(query, retrieve_chunk_ids(client, query, top_k)) for query in dataset.queries
    ]
    report = {
        "version": REPORT_VERSION,
        "run_id": run_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_dataset_file": DATASET_FILE,
        "evaluation_dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "api_base_url": base_url,
        "method": METHOD,
        "top_k": top_k,
        "ks": list(KS),
        "query_count": len(results),
        "aggregate": aggregate(results),
        "queries": results,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing run directory.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--top-k", type=int, default=max(KS), help="top_k sent to POST /query.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing evaluation_report.json."
    )
    args = parser.parse_args(argv)
    try:
        with httpx.Client(base_url=args.base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
            path = run_evaluation(args.run, client, args.base_url, args.top_k, args.overwrite)
    except EvaluationRunError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Evaluation report written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
