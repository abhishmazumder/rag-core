import pytest

from rag_core.application.chunking import fixed_size_chunks, recursive_chunks


@pytest.mark.parametrize("chunker", [fixed_size_chunks, recursive_chunks])
def test_chunkers_return_empty_for_empty_or_whitespace_input(chunker) -> None:
    assert chunker("", chunk_size_chars=5) == []
    assert chunker(" \n\t ", chunk_size_chars=5) == []


@pytest.mark.parametrize("chunker", [fixed_size_chunks, recursive_chunks])
def test_chunker_returns_one_chunk_when_short_or_exact_size(chunker) -> None:
    assert chunker("abc", chunk_size_chars=5) == ["abc"]
    assert chunker("abcde", chunk_size_chars=5) == ["abcde"]


@pytest.mark.parametrize("chunker", [fixed_size_chunks, recursive_chunks])
def test_chunkers_return_list_of_strings(chunker) -> None:
    chunks = chunker("abcdefghij", chunk_size_chars=4)

    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_fixed_size_chunks_split_at_character_limit() -> None:
    assert fixed_size_chunks("abcdefghij", chunk_size_chars=4) == ["abcd", "efgh", "ij"]


def test_fixed_size_chunks_preserve_exact_character_overlap() -> None:
    chunks = fixed_size_chunks("abcdefghij", chunk_size_chars=5, overlap_chars=2)

    assert chunks == ["abcde", "defgh", "ghij"]


@pytest.mark.parametrize("chunk_size_chars", [0, -1])
def test_chunkers_reject_nonpositive_chunk_size(chunk_size_chars: int) -> None:
    with pytest.raises(ValueError, match="chunk_size_chars must be greater than 0"):
        fixed_size_chunks("text", chunk_size_chars=chunk_size_chars)
    with pytest.raises(ValueError, match="chunk_size_chars must be greater than 0"):
        recursive_chunks("text", chunk_size_chars=chunk_size_chars)


@pytest.mark.parametrize("overlap_chars", [-1, 5])
def test_chunkers_reject_invalid_overlap(overlap_chars: int) -> None:
    with pytest.raises(ValueError, match="overlap_chars must be at least 0"):
        fixed_size_chunks("text", chunk_size_chars=5, overlap_chars=overlap_chars)
    with pytest.raises(ValueError, match="overlap_chars must be at least 0"):
        recursive_chunks("text", chunk_size_chars=5, overlap_chars=overlap_chars)


def test_chunkers_are_deterministic() -> None:
    first = recursive_chunks("alpha beta gamma delta", chunk_size_chars=10, overlap_chars=2)
    second = recursive_chunks("alpha beta gamma delta", chunk_size_chars=10, overlap_chars=2)

    assert first == second


def test_recursive_chunks_split_on_paragraph_boundaries() -> None:
    chunks = recursive_chunks("Alpha one. Beta two.\n\nGamma three.", chunk_size_chars=30)

    assert chunks == ["Alpha one. Beta two.\n\n", "Gamma three."]


def test_recursive_chunks_split_on_sentence_boundaries() -> None:
    chunks = recursive_chunks("Alpha one. Beta two.", chunk_size_chars=15)

    assert chunks == ["Alpha one. ", "Beta two."]


def test_recursive_chunks_split_on_whitespace_boundaries() -> None:
    chunks = recursive_chunks("alpha beta gamma", chunk_size_chars=10)

    assert chunks == ["alpha ", "beta gamma"]


def test_recursive_chunks_fall_back_to_hard_boundaries_and_apply_overlap() -> None:
    chunks = recursive_chunks("abcdefgh", chunk_size_chars=4, overlap_chars=2)

    assert chunks == ["abcd", "cdef", "efgh"]
