import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "evaluation" / "scripts" / "prepare_evaluation_corpus.py"
)
spec = importlib.util.spec_from_file_location("prepare_evaluation_corpus", SCRIPT)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

BASE_URL = "http://api.test"

SOURCE = {
    "version": "v1",
    "candidates": [
        {
            "candidate_id": "CAND-1",
            "documents": [
                {"document_key": "edu", "text": "education text"},
                {"document_key": "job", "text": "job text"},
            ],
        },
        {"candidate_id": "CAND-2", "documents": [{"document_key": "edu2", "text": "other text"}]},
    ],
}


def _write_source(tmp_path: Path, data: object = SOURCE) -> Path:
    path = tmp_path / "source.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _fake_api(requests: list[httpx.Request], fail_on: str | None = None) -> httpx.Client:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        if fail_on and body["text"] == fail_on:
            return httpx.Response(500, json={"detail": "boom"})
        counter["n"] += 1
        n = counter["n"]
        return httpx.Response(
            200,
            json={
                "candidate_id": body["candidate_id"],
                "document_id": f"doc-{n}",
                "chunks": [
                    {"id": f"chunk-{n}-{i}", "chunk_index": i, "text": f"{body['text']} #{i}"}
                    for i in range(2)
                ],
            },
        )

    return httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))


def _run(tmp_path, requests, source=SOURCE, run_id="run1", fail_on=None) -> Path:
    return prep.prepare_corpus(
        _write_source(tmp_path, source),
        BASE_URL,
        run_id=run_id,
        runs_dir=tmp_path / "runs",
        client=_fake_api(requests, fail_on),
    )


def test_parses_source_file(tmp_path) -> None:
    source = prep.load_source_documents(_write_source(tmp_path))

    assert [c.candidate_id for c in source.candidates] == ["CAND-1", "CAND-2"]
    assert [d.document_key for d in source.candidates[0].documents] == ["edu", "job"]


def test_posts_every_document_with_candidate_and_text(tmp_path) -> None:
    requests: list[httpx.Request] = []

    _run(tmp_path, requests)

    assert [r.method for r in requests] == ["POST"] * 3
    assert all(str(r.url) == f"{BASE_URL}/documents" for r in requests)
    assert [json.loads(r.content) for r in requests] == [
        {"candidate_id": "CAND-1", "text": "education text"},
        {"candidate_id": "CAND-1", "text": "job text"},
        {"candidate_id": "CAND-2", "text": "other text"},
    ]


def test_corpus_preserves_api_response_and_source_text(tmp_path) -> None:
    run_dir = _run(tmp_path, [])

    corpus = json.loads((run_dir / "corpus.json").read_text(encoding="utf-8"))

    assert corpus["version"] == "v1"
    assert corpus["source_documents_file"] == "source.json"
    assert [c["candidate_id"] for c in corpus["candidates"]] == ["CAND-1", "CAND-2"]
    assert corpus["candidates"][0]["documents"][0] == {
        "document_key": "edu",
        "document_id": "doc-1",
        "source_text": "education text",
        "chunks": [
            {"id": "chunk-1-0", "chunk_index": 0, "text": "education text #0"},
            {"id": "chunk-1-1", "chunk_index": 1, "text": "education text #1"},
        ],
    }
    assert len(corpus["candidates"][0]["documents"]) == 2
    assert corpus["candidates"][1]["documents"][0]["document_id"] == "doc-3"


def test_manifest_counts_are_computed(tmp_path) -> None:
    run_dir = _run(tmp_path, [])

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == "run1"
    assert manifest["source_documents_file"] == "source.json"
    assert manifest["api_base_url"] == BASE_URL
    assert manifest["candidate_count"] == 2
    assert manifest["document_count"] == 3
    assert manifest["chunk_count"] == 6
    assert "created_at" in manifest
    assert {p.name for p in run_dir.iterdir()} == {"corpus.json", "manifest.json"}


def _candidate(**overrides) -> dict:
    document = {"document_key": "k", "text": "t"}
    candidate = {"candidate_id": "c", "documents": [document]}
    return {"version": "v1", "candidates": [candidate | overrides]}


@pytest.mark.parametrize(
    "data",
    [
        {"candidates": SOURCE["candidates"]},
        {"version": "v1", "candidates": []},
        _candidate(candidate_id=""),
        _candidate(documents=[]),
        _candidate(documents=[{"document_key": "k", "text": ""}]),
        _candidate(documents=[{"text": "t"}]),
    ],
)
def test_invalid_source_fails_before_any_api_call(tmp_path, data) -> None:
    requests: list[httpx.Request] = []

    with pytest.raises(prep.CorpusPreparationError, match="Invalid source file"):
        _run(tmp_path, requests, source=data)

    assert requests == []
    assert not (tmp_path / "runs").exists()


def test_non_json_source_fails_clearly(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(prep.CorpusPreparationError, match="not valid JSON"):
        prep.load_source_documents(path)


def test_api_failure_fails_and_writes_nothing(tmp_path) -> None:
    with pytest.raises(prep.CorpusPreparationError, match="CAND-1/job"):
        _run(tmp_path, [], fail_on="job text")

    assert not (tmp_path / "runs" / "run1").exists()


def test_invalid_api_response_fails_clearly(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidate_id": "CAND-1", "document_id": "d"})

    client = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))

    with pytest.raises(prep.CorpusPreparationError, match="Invalid POST /documents response"):
        prep.prepare_corpus(
            _write_source(tmp_path), BASE_URL, run_id="r", runs_dir=tmp_path / "runs", client=client
        )


def test_existing_run_directory_is_not_overwritten(tmp_path) -> None:
    existing = tmp_path / "runs" / "run1"
    existing.mkdir(parents=True)
    (existing / "corpus.json").write_text("keep", encoding="utf-8")
    requests: list[httpx.Request] = []

    with pytest.raises(prep.CorpusPreparationError, match="already exists"):
        _run(tmp_path, requests)

    assert (existing / "corpus.json").read_text(encoding="utf-8") == "keep"
    assert requests == []


def test_generated_run_ids_are_unique(tmp_path) -> None:
    first = _run(tmp_path, [], run_id=None)
    second = _run(tmp_path, [], run_id=None)

    assert first != second
    assert first.exists() and second.exists()


def test_main_returns_error_code_on_failure(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")

    assert prep.main(["--source", str(path)]) == 1
    assert "error:" in capsys.readouterr().err
