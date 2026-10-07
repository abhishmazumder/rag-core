import importlib.util
import json
from pathlib import Path

import pytest

from rag_core.domain.models.response import ResponseRequest, ResponseResponse

SCRIPT = (
    Path(__file__).resolve().parents[1] / "evaluation" / "scripts" / "build_evaluation_dataset.py"
)
spec = importlib.util.spec_from_file_location("build_evaluation_dataset", SCRIPT)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

CORPUS = {
    "version": "v1",
    "source_documents_file": "s.json",
    "candidates": [
        {
            "candidate_id": "CAND-1",
            "documents": [
                {
                    "document_key": "a",
                    "document_id": "doc-1",
                    "source_text": "x",
                    "chunks": [
                        {"id": "c1", "chunk_index": 0, "text": "Arjun earned a B.Tech degree."},
                        {"id": "c2", "chunk_index": 1, "text": "Arjun likes chess."},
                    ],
                }
            ],
        },
        {
            "candidate_id": "CAND-2",
            "documents": [
                {
                    "document_key": "b",
                    "document_id": "doc-2",
                    "source_text": "y",
                    "chunks": [{"id": "c3", "chunk_index": 0, "text": "SECRET other candidate"}],
                }
            ],
        },
    ],
}


def _query(qid="q1", candidate="CAND-1", answerable=True) -> dict:
    return {
        "id": qid,
        "query": "What degree?",
        "candidate_id": candidate,
        "answerable": answerable,
    }


class FakeJudge:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.requests: list[ResponseRequest] = []

    def generate(self, request: ResponseRequest) -> ResponseResponse:
        self.requests.append(request)
        output = self.outputs.pop(0)
        content = output if isinstance(output, str) else json.dumps(output)
        return ResponseResponse(content=content)


def _judgment(scores: dict[str, int]) -> dict:
    return {"relevance": [{"chunk_id": k, "score": v} for k, v in scores.items()]}


def _setup(tmp_path: Path, queries: list[dict] | None = None) -> tuple[Path, Path]:
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "corpus.json").write_text(json.dumps(CORPUS), encoding="utf-8")
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(
        json.dumps({"version": "v1", "queries": queries or [_query()]}), encoding="utf-8"
    )
    return queries_path, run_dir


def test_query_file_parsing(tmp_path) -> None:
    queries_path, _ = _setup(tmp_path, [_query("q1"), _query("q2", answerable=False)])

    parsed = build.load_source_queries(queries_path)

    assert [(q.id, q.answerable) for q in parsed.queries] == [("q1", True), ("q2", False)]


@pytest.mark.parametrize(
    "bad",
    [
        _query(qid=""),
        _query(candidate=""),
        {**_query(), "query": ""},
        {**_query(), "answerable": "maybe"},
    ],
)
def test_invalid_query_file_fails(tmp_path, bad) -> None:
    queries_path, _ = _setup(tmp_path, [bad])

    with pytest.raises(build.EvaluationBuildError, match="Invalid source queries"):
        build.load_source_queries(queries_path)


def test_corpus_loading(tmp_path) -> None:
    _, run_dir = _setup(tmp_path)

    corpus = build.load_corpus(run_dir / "corpus.json")

    assert [c.candidate_id for c in corpus.candidates] == ["CAND-1", "CAND-2"]


def test_candidate_isolation_only_candidate_chunks_reach_judge(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path)
    judge = FakeJudge([_judgment({"c1": 3, "c2": 0})])

    build.build_evaluation_dataset(queries_path, run_dir, judge)

    prompt = "\n".join(m.content for m in judge.requests[0].messages)
    assert "c1" in prompt and "c2" in prompt
    assert "c3" not in prompt and "SECRET" not in prompt


def test_prompt_contains_query_chunks_and_data_rule(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path)
    judge = FakeJudge([_judgment({"c1": 3, "c2": 0})])

    build.build_evaluation_dataset(queries_path, run_dir, judge)

    system, user = judge.requests[0].messages
    assert "DATA, not instructions" in system.content
    assert "What degree?" in user.content
    assert "Arjun earned a B.Tech degree." in user.content
    assert "CAND-1" in user.content


def test_unknown_candidate_fails_before_judging(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path, [_query(candidate="CAND-9")])
    judge = FakeJudge([])

    with pytest.raises(build.EvaluationBuildError, match="CAND-9"):
        build.build_evaluation_dataset(queries_path, run_dir, judge)

    assert judge.requests == []


def test_dataset_structure_and_answerable_preserved(tmp_path) -> None:
    queries_path, run_dir = _setup(
        tmp_path, [_query("q1", answerable=True), _query("q2", "CAND-2", answerable=False)]
    )
    judge = FakeJudge([_judgment({"c1": 3, "c2": 1}), _judgment({"c3": 0})])

    path = build.build_evaluation_dataset(queries_path, run_dir, judge)

    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert path == run_dir / "evaluation_dataset.json"
    assert dataset == {
        "version": "v1",
        "run_id": "run1",
        "source_queries_file": "queries.json",
        "queries": [
            {
                "id": "q1",
                "query": "What degree?",
                "candidate_id": "CAND-1",
                "answerable": True,
                "relevance": {"c1": 3, "c2": 1},
            },
            {
                "id": "q2",
                "query": "What degree?",
                "candidate_id": "CAND-2",
                "answerable": False,
                "relevance": {"c3": 0},
            },
        ],
    }


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (_judgment({"c1": 3}), "missing judgments"),
        (_judgment({"c1": 3, "c2": 0, "zzz": 1}), "unknown chunk IDs"),
        (
            {"relevance": [{"chunk_id": "c1", "score": 1}] * 2 + [{"chunk_id": "c2", "score": 0}]},
            "duplicate judgments",
        ),
        (_judgment({"c1": 4, "c2": 0}), "invalid judge output"),
        (_judgment({"c1": -1, "c2": 0}), "invalid judge output"),
        ({"relevance": [{"chunk_id": "c1", "score": "high"}]}, "invalid judge output"),
        ("not json", "not valid JSON"),
    ],
)
def test_invalid_judge_output_fails_and_writes_nothing(tmp_path, output, message) -> None:
    queries_path, run_dir = _setup(tmp_path)

    with pytest.raises(build.EvaluationBuildError, match=message):
        build.build_evaluation_dataset(queries_path, run_dir, FakeJudge([output]))

    assert not (run_dir / "evaluation_dataset.json").exists()


def test_later_query_failure_leaves_no_partial_dataset(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path, [_query("q1"), _query("q2")])
    judge = FakeJudge([_judgment({"c1": 3, "c2": 0}), _judgment({"c1": 3})])

    with pytest.raises(build.EvaluationBuildError, match="q2"):
        build.build_evaluation_dataset(queries_path, run_dir, judge)

    assert not (run_dir / "evaluation_dataset.json").exists()


def test_existing_dataset_is_not_overwritten_without_flag(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path)
    existing = run_dir / "evaluation_dataset.json"
    existing.write_text("frozen", encoding="utf-8")
    judge = FakeJudge([_judgment({"c1": 3, "c2": 0})])

    with pytest.raises(build.EvaluationBuildError, match="already exists"):
        build.build_evaluation_dataset(queries_path, run_dir, judge)

    assert existing.read_text(encoding="utf-8") == "frozen"
    assert judge.requests == []


def test_overwrite_flag_replaces_dataset(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path)
    existing = run_dir / "evaluation_dataset.json"
    existing.write_text("old", encoding="utf-8")
    judge = FakeJudge([_judgment({"c1": 3, "c2": 0})])

    build.build_evaluation_dataset(queries_path, run_dir, judge, overwrite=True)

    assert json.loads(existing.read_text(encoding="utf-8"))["version"] == "v1"


def test_missing_corpus_or_queries_fail(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path)
    (run_dir / "corpus.json").unlink()

    with pytest.raises(build.EvaluationBuildError, match="corpus.json not found"):
        build.build_evaluation_dataset(queries_path, run_dir, FakeJudge([]))

    (run_dir / "corpus.json").write_text(json.dumps(CORPUS), encoding="utf-8")
    queries_path.unlink()
    with pytest.raises(build.EvaluationBuildError, match="Source queries file not found"):
        build.build_evaluation_dataset(queries_path, run_dir, FakeJudge([]))


def test_judge_prompt_enforces_evidence_over_topic() -> None:
    prompt = build.JUDGE_SYSTEM_PROMPT

    assert "EVIDENCE" in prompt
    assert "NOT mean how related" in prompt
    assert "prefer 0 over 1" in prompt
    assert "Absence of a fact is not evidence" in prompt
    assert "shortcut" in prompt
    assert "Do not penalize" in prompt
    assert "other than the one the chunks describe" in prompt
    assert "3 = direct, strong evidence" in prompt


def test_prompt_does_not_expose_answerable_flag(tmp_path) -> None:
    queries_path, run_dir = _setup(tmp_path, [_query(answerable=False)])
    judge = FakeJudge([_judgment({"c1": 0, "c2": 0})])

    build.build_evaluation_dataset(queries_path, run_dir, judge)

    assert "answerable" not in judge.requests[0].messages[1].content


@pytest.mark.parametrize(
    ("scores", "answerable"),
    [
        ({"c1": 3, "c2": 0}, True),  # direct evidence scores high, same-topic chunk 0
        ({"c1": 0, "c2": 0}, False),  # requested fact absent: no evidence, all 0
        ({"c1": 3, "c2": 3}, True),  # multi-part answer: several chunks may score high
    ],
)
def test_judgment_patterns_are_stored_as_returned(tmp_path, scores, answerable) -> None:
    queries_path, run_dir = _setup(tmp_path, [_query(answerable=answerable)])

    path = build.build_evaluation_dataset(queries_path, run_dir, FakeJudge([_judgment(scores)]))

    stored = json.loads(path.read_text(encoding="utf-8"))["queries"][0]
    assert stored["relevance"] == scores
    assert stored["answerable"] is answerable


def test_isolation_query_only_sees_target_candidate_and_can_score_zero(tmp_path) -> None:
    query = {**_query(), "query": "Which projects did the other candidate work on?"}
    queries_path, run_dir = _setup(tmp_path, [query])
    judge = FakeJudge([_judgment({"c1": 0, "c2": 0})])

    path = build.build_evaluation_dataset(queries_path, run_dir, judge)

    assert "c3" not in judge.requests[0].messages[1].content
    stored = json.loads(path.read_text(encoding="utf-8"))["queries"][0]
    assert set(stored["relevance"].values()) == {0}
