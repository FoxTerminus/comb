"""Synthetic smoke/dev examples for validating benchmark plumbing."""

from __future__ import annotations

from benchmarks.scripts.schema import BenchmarkExample


def _context_length(text: str) -> int:
    return len(text.split())


def _example(
    *,
    id: str,
    benchmark: str,
    task: str,
    split: str,
    context: str,
    question: str,
    answer: str,
    metric: str,
    **metadata: object,
) -> BenchmarkExample:
    return BenchmarkExample(
        id=id,
        benchmark=benchmark,
        task=task,
        split=split,
        context=context,
        question=question,
        answer=answer,
        choices=metadata.pop("choices", None),  # type: ignore[arg-type]
        metadata={
            "context_length": _context_length(context),
            "source": "synthetic",
            "metric": metric,
            **metadata,
        },
    )


def build_synthetic_examples(split: str) -> list[BenchmarkExample]:
    repeat = 1 if split == "smoke" else 10
    examples: list[BenchmarkExample] = []
    for idx in range(repeat):
        suffix = "" if split == "smoke" else f"-{idx:02d}"
        filler = " ".join([f"distractor{idx}_{n}" for n in range(20 if split == "smoke" else 120)])

        examples.append(
            _example(
                id=f"ruler-needle{suffix}",
                benchmark="RULER",
                task="needle_retrieval",
                split=split,
                context=f"{filler} The secret code is ORCHID-{idx}. {filler}",
                question="What is the secret code?",
                answer=f"ORCHID-{idx}",
                metric="exact_match",
                task_type="retrieval",
                needle_position="middle",
            )
        )
        examples.append(
            _example(
                id=f"ruler-multihop{suffix}",
                benchmark="RULER",
                task="multi_hop_trace",
                split=split,
                context=(
                    f"Alice points to Bob. Bob points to Clara. Clara keeps token LANTERN-{idx}. "
                    f"{filler}"
                ),
                question="Which token is kept by the final person in the chain?",
                answer=f"LANTERN-{idx}",
                metric="contains",
                task_type="multi_hop",
            )
        )
        examples.append(
            _example(
                id=f"scbench-shared{suffix}",
                benchmark="SCBench",
                task="shared_context_retrieval",
                split=split,
                context=f"Shared project notes: cache key alpha maps to value NEBULA-{idx}. {filler}",
                question="In the shared project notes, what value does cache key alpha map to?",
                answer=f"NEBULA-{idx}",
                metric="exact_match",
                shared_context_id=f"shared-{idx}",
                query_index=0,
                cache_reuse=True,
            )
        )
        examples.append(
            _example(
                id=f"scbench-semantic{suffix}",
                benchmark="SCBench",
                task="semantic_retrieval",
                split=split,
                context=f"The meeting used a nautical codename. The codename was HARBOR-{idx}. {filler}",
                question="What was the nautical codename?",
                answer=f"HARBOR-{idx}",
                metric="contains",
                shared_context_id=f"semantic-{idx}",
                query_index=1,
                cache_reuse=True,
            )
        )
        examples.append(
            _example(
                id=f"longbench-qa{suffix}",
                benchmark="LongBench",
                task="single_doc_qa",
                split=split,
                context=f"Article: The observatory opened in 1998. Its director is Mira Chen. {filler}",
                question="Who is the director of the observatory?",
                answer="Mira Chen",
                metric="f1",
                official_task="qasper",
                language="en",
                length_bucket="short" if split == "smoke" else "medium",
            )
        )
        examples.append(
            _example(
                id=f"longbench-classification{suffix}",
                benchmark="LongBench",
                task="classification",
                split=split,
                context=f"Review: The tool was reliable, fast, and pleasant to use. {filler}",
                question="Classify the sentiment as positive or negative.",
                answer="positive",
                metric="classification",
                choices=["positive", "negative"],
                official_task="trec",
                language="en",
                length_bucket="short",
            )
        )
        examples.append(
            _example(
                id=f"locomo-qa{suffix}",
                benchmark="LoCoMo",
                task="text_qa",
                split=split,
                context=(
                    f"Session 1: Dana said her birthday dinner is at Cedar Hall. "
                    f"Session 2: Dana changed the music playlist only. {filler}"
                ),
                question="Where is Dana's birthday dinner?",
                answer="Cedar Hall",
                metric="f1",
                session_id=f"dialogue-{idx}",
                turn=2,
                question_type="single_hop",
                temporal=False,
            )
        )
        examples.append(
            _example(
                id=f"longcodebench-codeqa{suffix}",
                benchmark="LongCodeBench",
                task="code_qa",
                split=split,
                context=(
                    "File utils.py defines def normalize_name(value): return value.strip().lower(). "
                    "File app.py calls normalize_name before storing user names. "
                    f"{filler}"
                ),
                question="Which function normalizes user names before storage?",
                answer="normalize_name",
                metric="code",
                repo=f"synthetic-repo-{idx}",
                file="app.py",
                language="python",
                task_type="code_qa",
            )
        )
    return examples

