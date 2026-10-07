"""Ingest evaluation source documents through POST /documents and snapshot the result.

This is an external HTTP client of the API; it imports nothing from rag_core.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "runs"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
CORPUS_VERSION = "v1"
REQUEST_TIMEOUT_SECONDS = 120.0


class CorpusPreparationError(Exception):
    pass


class SourceDocument(BaseModel):
    document_key: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SourceCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    documents: list[SourceDocument] = Field(min_length=1)


class SourceDocumentsFile(BaseModel):
    version: str = Field(min_length=1)
    candidates: list[SourceCandidate] = Field(min_length=1)


class IngestedChunk(BaseModel):
    id: str = Field(min_length=1)
    chunk_index: int
    text: str


class IngestResponse(BaseModel):
    candidate_id: str
    document_id: str = Field(min_length=1)
    chunks: list[IngestedChunk]


def load_source_documents(path: Path) -> SourceDocumentsFile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return SourceDocumentsFile.model_validate(raw)
    except OSError as error:
        raise CorpusPreparationError(f"Cannot read source file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CorpusPreparationError(f"Source file {path} is not valid JSON: {error}") from error
    except ValidationError as error:
        raise CorpusPreparationError(f"Invalid source file {path}:\n{error}") from error


def ingest_document(
    client: httpx.Client, candidate_id: str, document: SourceDocument
) -> IngestResponse:
    label = f"{candidate_id}/{document.document_key}"
    try:
        response = client.post(
            "/documents", json={"candidate_id": candidate_id, "text": document.text}
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise CorpusPreparationError(f"POST /documents failed for {label}: {error}") from error
    try:
        ingested = IngestResponse.model_validate(response.json())
    except (ValueError, ValidationError) as error:
        raise CorpusPreparationError(
            f"Invalid POST /documents response for {label}: {error}"
        ) from error
    if ingested.candidate_id != candidate_id:
        raise CorpusPreparationError(
            f"POST /documents returned candidate_id {ingested.candidate_id!r} "
            f"for {label}; expected {candidate_id!r}"
        )
    return ingested


def build_corpus(
    source: SourceDocumentsFile, source_name: str, client: httpx.Client
) -> dict[str, object]:
    candidates = []
    for candidate in source.candidates:
        documents = []
        for document in candidate.documents:
            ingested = ingest_document(client, candidate.candidate_id, document)
            documents.append(
                {
                    "document_key": document.document_key,
                    "document_id": ingested.document_id,
                    "source_text": document.text,
                    "chunks": [chunk.model_dump() for chunk in ingested.chunks],
                }
            )
        candidates.append({"candidate_id": candidate.candidate_id, "documents": documents})
    return {
        "version": CORPUS_VERSION,
        "source_documents_file": source_name,
        "candidates": candidates,
    }


def build_manifest(
    run_id: str, created_at: str, source_name: str, base_url: str, corpus: dict[str, object]
) -> dict[str, object]:
    candidates = corpus["candidates"]
    documents = [doc for candidate in candidates for doc in candidate["documents"]]
    return {
        "run_id": run_id,
        "created_at": created_at,
        "source_documents_file": source_name,
        "api_base_url": base_url,
        "candidate_count": len(candidates),
        "document_count": len(documents),
        "chunk_count": sum(len(doc["chunks"]) for doc in documents),
    }


def _create_run_directory(runs_dir: Path, run_id: str | None) -> tuple[str, Path]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        candidates = [run_id]
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        candidates = [stamp] + [f"{stamp}_{n}" for n in range(1, 100)]
    for candidate in candidates:
        run_dir = runs_dir / candidate
        try:
            run_dir.mkdir()
        except FileExistsError:
            if run_id is not None:
                raise CorpusPreparationError(f"Run directory already exists: {run_dir}") from None
            continue
        return candidate, run_dir
    raise CorpusPreparationError("Could not allocate a unique run directory.")


def prepare_corpus(
    source_path: Path,
    base_url: str,
    run_id: str | None = None,
    runs_dir: Path = RUNS_DIR,
    client: httpx.Client | None = None,
) -> Path:
    source = load_source_documents(source_path)
    if run_id is not None and (runs_dir / run_id).exists():
        raise CorpusPreparationError(f"Run directory already exists: {runs_dir / run_id}")
    owns_client = client is None
    http_client = client or httpx.Client(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        corpus = build_corpus(source, source_path.name, http_client)
    finally:
        if owns_client:
            http_client.close()

    created_at = datetime.now(UTC).isoformat()
    final_run_id, run_dir = _create_run_directory(runs_dir, run_id)
    manifest = build_manifest(final_run_id, created_at, source_path.name, base_url, corpus)
    (run_dir / "corpus.json").write_text(json.dumps(corpus, indent=2) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source documents JSON file.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--run-id", default=None, help="Explicit run ID (must not already exist).")
    args = parser.parse_args(argv)
    try:
        run_dir = prepare_corpus(args.source, args.base_url, args.run_id)
    except CorpusPreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Corpus written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
