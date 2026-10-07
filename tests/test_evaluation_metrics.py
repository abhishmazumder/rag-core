import importlib.util
import json
import math
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import retrieval_metrics as m  # noqa: E402

spec = importlib.util.spec_from_file_location("run_evaluation", SCRIPTS / "run_evaluation.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

LOG2_3 = math.log2(3)


def test_precision_all_relevant() -> None:
    assert m.precision_at_k([3, 1, 2, 3], 4) == 1.0


def test_precision_none_relevant() -> None:
    assert m.precision_at_k([0, 0, 0, 0], 4) == 0.0


def test_precision_partial() -> None:
    assert m.precision_at_k([3, 0, 1, 0], 4) == 0.5
    assert m.precision_at_k([3, 0, 1, 0], 1) == 1.0
    assert m.precision_at_k([3, 0, 1, 0], 2) == 0.5


def test_precision_divides_by_k_when_fewer_results_returned() -> None:
    assert m.precision_at_k([3], 4) == 0.25


def test_binary_relevance_for_precision_recall_mrr() -> None:
    assert m.precision_at_k([1, 3], 2) == 1.0
    assert m.recall_at_k([1], 1, 1) == 1.0
    assert m.reciprocal_rank([1]) == 1.0


def test_recall_all_relevant_retrieved() -> None:
    assert m.recall_at_k([3, 2, 0, 0], 2, 2) == 1.0


def test_recall_partial_retrieval() -> None:
    assert m.recall_at_k([3, 0, 0, 0], 2, 4) == 0.5
    assert m.recall_at_k([3, 0, 2, 0], 3, 2) == pytest.approx(1 / 3)


def test_recall_undefined_without_relevant_ground_truth() -> None:
    assert m.recall_at_k([0, 0, 0, 0], 0, 4) is None


def test_mrr_first_result_relevant() -> None:
    assert m.reciprocal_rank([3, 0, 0, 0]) == 1.0


def test_mrr_second_result_relevant() -> None:
    assert m.reciprocal_rank([0, 2, 3, 0]) == 0.5


def test_mrr_no_relevant_result() -> None:
    assert m.reciprocal_rank([0, 0, 0, 0]) == 0.0
    assert m.reciprocal_rank([]) == 0.0


def test_ndcg_perfect_ranking() -> None:
    assert m.ndcg_at_k([3, 2, 0, 0], [3, 2, 0, 0], 4) == pytest.approx(1.0)


def test_ndcg_reversed_ranking() -> None:
    # DCG = 3/log2(3) + 7/log2(3)... see below; hand-derived for k=2:
    # retrieved [2, 3]: DCG = 3/1 + 7/log2(3); ideal [3, 2]: IDCG = 7/1 + 3/log2(3)
    expected = (3 + 7 / LOG2_3) / (7 + 3 / LOG2_3)
    assert m.ndcg_at_k([2, 3], [3, 2], 2) == pytest.approx(expected)
    assert expected < 1.0


def test_ndcg_partially_relevant_ranking() -> None:
    # retrieved [0, 3]: DCG = 7/log2(3); ideal [3, 0]: IDCG = 7 -> 1/log2(3)
    assert m.ndcg_at_k([0, 3], [3, 0], 2) == pytest.approx(1 / LOG2_3)


def test_ndcg_preserves_graded_relevance() -> None:
    # gain (2^1 - 1) = 1 versus ideal (2^3 - 1) = 7, not binary 1/1
    assert m.ndcg_at_k([1], [3], 1) == pytest.approx(1 / 7)


def test_ndcg_undefined_when_ideal_dcg_is_zero() -> None:
    assert m.ndcg_at_k([0, 0, 0, 0], [0, 0, 0, 0], 4) is None


def test_aggregate_mean_ignores_none() -> None:
    assert m.aggregate_mean([1.0, None, 0.0]) == 0.5
    assert m.aggregate_mean([None, None]) is None


def _query(relevance: dict[str, int]):
    return run.DatasetQuery(
        id="q1", query="x", candidate_id="CAND-1", answerable=True, relevance=relevance
    )


def test_score_query_preserves_retrieval_order_and_maps_scores() -> None:
    query = _query({"a": 3, "b": 0, "c": 2, "d": 0})

    result = run.score_query(query, ["c", "b", "a", "d"])

    assert result["retrieved_chunk_ids"] == ["c", "b", "a", "d"]
    assert result["relevance_scores_in_retrieval_order"] == [2, 0, 3, 0]
    assert result["precision_at_2"] == 0.5
    assert result["recall_at_1"] == 0.5
    assert result["reciprocal_rank"] == 1.0
    assert result["ndcg_at_1"] == pytest.approx(3 / 7)
    assert "embedding" not in result


def test_score_query_unknown_chunk_id_fails() -> None:
    with pytest.raises(run.EvaluationRunError, match="not in the evaluation dataset"):
        run.score_query(_query({"a": 3}), ["a", "zzz"])


def test_score_query_without_relevant_chunks_has_undefined_recall_and_ndcg() -> None:
    result = run.score_query(_query({"a": 0, "b": 0}), ["a", "b"])

    assert result["recall_at_4"] is None
    assert result["ndcg_at_4"] is None
    assert result["precision_at_4"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def _dataset(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    dataset = {
        "version": "v1",
        "run_id": "run1",
        "source_queries_file": "q.json",
        "queries": [
            {
                "id": "q1",
                "query": "degree?",
                "candidate_id": "CAND-1",
                "answerable": True,
                "relevance": {"a": 3, "b": 0, "c": 0, "d": 0},
            },
            {
                "id": "q2",
                "query": "salary?",
                "candidate_id": "CAND-1",
                "answerable": False,
                "relevance": {"a": 0, "b": 0, "c": 0, "d": 0},
            },
        ],
    }
    (run_dir / "evaluation_dataset.json").write_text(json.dumps(dataset), encoding="utf-8")
    return run_dir


def _client(requests: list[dict], ids: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"chunks": [{"id": cid, "text": "t"} for cid in ids]})

    return httpx.Client(base_url="http://api.test", transport=httpx.MockTransport(handler))


def test_run_evaluation_calls_query_and_writes_report(tmp_path) -> None:
    run_dir = _dataset(tmp_path)
    dataset_before = (run_dir / "evaluation_dataset.json").read_bytes()
    requests: list[dict] = []

    path = run.run_evaluation(
        run_dir, _client(requests, ["b", "a", "c", "d"]), "http://api.test", 4
    )

    assert requests[0] == {
        "query": "degree?",
        "candidate_id": "CAND-1",
        "method": "vector",
        "top_k": 4,
    }
    report = json.loads(path.read_text(encoding="utf-8"))
    assert path == run_dir / "evaluation_report.json"
    assert report["query_count"] == 2
    assert report["method"] == "vector"
    assert report["queries"][0]["reciprocal_rank"] == 0.5
    assert report["aggregate"]["mean"]["reciprocal_rank"] == 0.25
    assert report["aggregate"]["mean"]["recall_at_4"] == 1.0
    assert report["aggregate"]["queries_with_defined_value"]["recall_at_4"] == 1
    assert (run_dir / "evaluation_dataset.json").read_bytes() == dataset_before


def test_run_evaluation_fails_on_unknown_chunk_and_writes_nothing(tmp_path) -> None:
    run_dir = _dataset(tmp_path)

    with pytest.raises(run.EvaluationRunError, match="not in the evaluation dataset"):
        run.run_evaluation(run_dir, _client([], ["a", "zzz"]), "http://api.test", 4)

    assert not (run_dir / "evaluation_report.json").exists()


def test_run_evaluation_does_not_overwrite_report_without_flag(tmp_path) -> None:
    run_dir = _dataset(tmp_path)
    report = run_dir / "evaluation_report.json"
    report.write_text("keep", encoding="utf-8")
    requests: list[dict] = []

    with pytest.raises(run.EvaluationRunError, match="already exists"):
        run.run_evaluation(run_dir, _client(requests, ["a"]), "http://api.test", 4)

    assert report.read_text(encoding="utf-8") == "keep"
    assert requests == []
    run.run_evaluation(run_dir, _client([], ["a"]), "http://api.test", 4, overwrite=True)
    assert json.loads(report.read_text(encoding="utf-8"))["query_count"] == 2


def test_run_evaluation_api_failure_fails_clearly(tmp_path) -> None:
    run_dir = _dataset(tmp_path)
    client = httpx.Client(
        base_url="http://api.test",
        transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"detail": "x"})),
    )

    with pytest.raises(run.EvaluationRunError, match="POST /query failed for q1"):
        run.run_evaluation(run_dir, client, "http://api.test", 4)


def test_top_k_below_four_is_rejected(tmp_path) -> None:
    with pytest.raises(run.EvaluationRunError, match="--top-k"):
        run.run_evaluation(_dataset(tmp_path), _client([], []), "http://api.test", 2)
