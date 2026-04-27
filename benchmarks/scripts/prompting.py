"""Prompt rendering and CombLlama input packing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from benchmarks.scripts.schema import BenchmarkExample


LONGBENCH_SUMMARY_TASKS = {
    "gov_report",
    "gov_report_e",
    "qmsum",
    "multi_news",
    "multi_news_e",
    "vcsum",
    "samsum",
    "samsum_e",
}
LONGBENCH_CODE_TASKS = {"lcc", "lcc_e", "repobench-p", "repobench-p_e"}
LONGBENCH_RETRIEVAL_TASKS = {
    "passage_count",
    "passage_count_e",
    "passage_retrieval_en",
    "passage_retrieval_en_e",
    "passage_retrieval_zh",
}


def _lettered_choices(choices: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + idx)}) {choice}" for idx, choice in enumerate(choices))


def render_qa_prompt(example: BenchmarkExample) -> str:
    if example.benchmark == "RULER":
        return (
            "Find the exact secret passkey in the context. "
            "Return only the passkey string, with no explanation.\n\n"
            f"Context:\n{example.context}\n\n"
            f"Question:\n{example.question}\n\n"
            "Passkey:"
        )

    if example.choices:
        if example.benchmark == "LoCoMo":
            return (
                "Answer the long-dialogue memory question using only the provided conversation. "
                "Choose from the options and return only the exact answer text, not the option letter. "
                "Do not explain.\n\n"
                f"Conversation:\n{example.context}\n\n"
                f"Question:\n{example.question}\n\n"
                f"Options:\n{_lettered_choices(example.choices)}\n\n"
                "Answer text:"
            )
        return (
            "Answer the multiple-choice question using only the provided context. "
            "Return only the option letter, for example A, B, C, and so on. "
            "Do not explain.\n\n"
            f"Context:\n{example.context}\n\n"
            f"Question:\n{example.question}\n\n"
            f"Choices:\n{_lettered_choices(example.choices)}\n\n"
            "Option letter:"
        )

    if example.benchmark == "LongCodeBench" or (
        example.benchmark == "LongBench" and example.task in LONGBENCH_CODE_TASKS
    ):
        return (
            "Use the repository/code context to answer the coding task. "
            "If this is a completion task, return only the missing code. "
            "If this is a code question, return only the answer.\n\n"
            f"Code context:\n{example.context}\n\n"
            f"Task:\n{example.question}\n\n"
            "Answer:"
        )

    if example.benchmark == "LongBench" and example.task in LONGBENCH_SUMMARY_TASKS:
        return (
            "Read the provided long context and write a concise answer or summary "
            "that directly satisfies the request. Do not include preambles.\n\n"
            f"Context:\n{example.context}\n\n"
            f"Request:\n{example.question}\n\n"
            "Summary:"
        )

    if example.benchmark == "LongBench" and example.task in LONGBENCH_RETRIEVAL_TASKS:
        return (
            "Answer the retrieval/counting question using only the provided context. "
            "Return only the exact requested label, title, or number.\n\n"
            f"Context:\n{example.context}\n\n"
            f"Question:\n{example.question}\n\n"
            "Answer:"
        )

    if example.benchmark == "SCBench":
        if example.task == "scbench_summary":
            instruction = (
                "Summarize the provided context according to the question. "
                "Return only the concise summary."
            )
        elif example.task in {"scbench_repoqa", "scbench_repoqa_and_kv"}:
            instruction = (
                "Answer the repository/code question using only the code context. "
                "Return only the requested code or identifier."
            )
        elif example.task in {"scbench_kv", "scbench_prefix_suffix", "scbench_mf"}:
            instruction = (
                "Retrieve the exact value requested from the context. "
                "Return only the value, with no explanation."
            )
        else:
            instruction = (
                "Answer the question using only the provided long context. "
                "Return a short direct answer."
            )
        return (
            f"{instruction}\n\n"
            f"Context:\n{example.context}\n\n"
            f"Question:\n{example.question}\n\n"
            "Answer:"
        )

    choices = ""
    if example.choices:
        choices = "\n\nChoices:\n" + "\n".join(
            f"{idx + 1}. {choice}" for idx, choice in enumerate(example.choices)
        )
    return (
        "You are evaluating long-context understanding. Answer the question "
        "using only the provided context. Keep the answer short and direct.\n\n"
        f"Context:\n{example.context}\n\n"
        f"Question:\n{example.question}"
        f"{choices}\n\n"
        "Answer:"
    )


@dataclass
class PackedCombInput:
    tensors: dict[str, Any]
    prompt_tokens: int
    chunk_tokens: int
    decoder_tokens: int


@dataclass(frozen=True)
class PackingDiagnostics:
    prompt_tokens: int
    history_tokens: int
    decoder_tokens: int
    chunk_size: int
    total_chunks: int
    kept_chunks: int
    dropped_chunks: int
    kept_chunk_indices: list[int]
    kept_chunk_tokens: int
    dropped_chunk_tokens: int
    compression_ratio: float
    retention_policy: str


def _chunk_tokens(tokens: list[int], chunk_size: int) -> list[list[int]]:
    return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]


def _select_chunks(
    chunks: list[list[int]],
    *,
    compression_ratio: float,
    retention_policy: str,
    answer_token_start: int | None,
    chunk_size: int,
    history_tokens: int,
) -> list[list[int]]:
    del compression_ratio, answer_token_start, chunk_size, history_tokens
    if retention_policy != "all_encoder_chunks":
        raise ValueError(
            "Unsupported retention_policy: "
            f"{retention_policy}. Only all_encoder_chunks is supported."
        )
    if not chunks:
        return chunks
    return chunks


def inspect_combllama_packing(
    token_ids: list[int],
    *,
    chunk_size: int,
    recent_window_tokens: int,
    compression_ratio: float,
    retention_policy: str = "all_encoder_chunks",
    answer_token_start: int | None = None,
) -> PackingDiagnostics:
    """Return token/chunk retention metadata for the current packing policy."""

    del answer_token_start
    if retention_policy != "all_encoder_chunks":
        raise ValueError(
            "Unsupported retention_policy: "
            f"{retention_policy}. Only all_encoder_chunks is supported."
        )
    if not token_ids:
        raise ValueError("Cannot inspect an empty prompt")

    chunk_size = max(1, chunk_size)
    recent_window_tokens = max(1, recent_window_tokens)
    decoder_tokens = min(len(token_ids), recent_window_tokens)
    history_tokens = max(0, len(token_ids) - decoder_tokens)
    chunks = _chunk_tokens(token_ids[:history_tokens], chunk_size)
    total_chunks = len(chunks)
    kept_chunks = total_chunks
    indices = list(range(total_chunks))
    kept = [chunks[idx] for idx in indices]
    kept_chunk_tokens = sum(len(chunk) for chunk in kept)
    dropped_chunk_tokens = history_tokens - kept_chunk_tokens

    return PackingDiagnostics(
        prompt_tokens=len(token_ids),
        history_tokens=history_tokens,
        decoder_tokens=decoder_tokens,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
        kept_chunks=kept_chunks,
        dropped_chunks=total_chunks - kept_chunks,
        kept_chunk_indices=indices,
        kept_chunk_tokens=kept_chunk_tokens,
        dropped_chunk_tokens=dropped_chunk_tokens,
        compression_ratio=compression_ratio,
        retention_policy=retention_policy,
    )


def pack_combllama_prompt(
    token_ids: list[int],
    *,
    chunk_size: int,
    recent_window_tokens: int,
    compression_ratio: float,
    retention_policy: str = "all_encoder_chunks",
    answer_token_start: int | None = None,
) -> PackedCombInput:
    """Pack a plain prompt into CombLlama chunk context plus decoder input.

    CombLlama consumes old context through `chunk_ids` and keeps the most recent
    tokens in the decoder path. The `all_encoder_chunks` policy sends every
    history chunk through the chunk encoder, so compression comes from the
    CombLlama encoder/decoder architecture rather than benchmark-side chunk
    dropping.
    """

    if not token_ids:
        raise ValueError("Cannot pack an empty prompt")

    recent_window_tokens = max(1, recent_window_tokens)
    decoder_ids = token_ids[-recent_window_tokens:]
    history_ids = token_ids[: max(0, len(token_ids) - len(decoder_ids))]

    chunk_size = max(1, chunk_size)
    chunks = _chunk_tokens(history_ids, chunk_size)
    chunks = _select_chunks(
        chunks,
        compression_ratio=compression_ratio,
        retention_policy=retention_policy,
        answer_token_start=answer_token_start,
        chunk_size=chunk_size,
        history_tokens=len(history_ids),
    )

    tensors: dict[str, Any] = {
        "input_ids": torch.tensor([decoder_ids], dtype=torch.long),
        "position_ids": torch.arange(len(decoder_ids), dtype=torch.long).unsqueeze(0),
        "cu_seqlens_q": torch.tensor([0, len(decoder_ids)], dtype=torch.int32),
        "max_seqlen_q": len(decoder_ids),
    }

    if chunks:
        flat_chunks = [token for chunk in chunks for token in chunk]
        cu_chunk = [0]
        pos_k = []
        for chunk in chunks:
            cu_chunk.append(cu_chunk[-1] + len(chunk))
            pos_k.extend(range(len(chunk)))

        tensors.update(
            {
                "chunk_ids": torch.tensor([flat_chunks], dtype=torch.long),
                "position_ids_k": torch.tensor([pos_k], dtype=torch.long),
                "cu_seqlens_k": torch.tensor([0, len(flat_chunks)], dtype=torch.int32),
                "max_seqlen_k": len(flat_chunks),
                "cu_seqlens_chunk": torch.tensor(cu_chunk, dtype=torch.int32),
                "max_seqlen_chunk": max(len(chunk) for chunk in chunks),
            }
        )

    return PackedCombInput(
        tensors=tensors,
        prompt_tokens=len(token_ids),
        chunk_tokens=sum(len(chunk) for chunk in chunks),
        decoder_tokens=len(decoder_ids),
    )
