"""Synthetic Phonebook data generation.

The SambaY paper evaluates Phonebook as a realistic multi-key-value
retrieval task with 32K context and about 1,850 name-number pairs. This
module follows that shape with deterministic synthetic entries.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable


SYSTEM_PROMPT = (
    "You are a retrieval assistant. Answer using only the provided phonebook. "
    "Return only the requested phone number."
)

FIRST_NAMES = [
    "Alden", "Beatrice", "Calvin", "Daria", "Elias", "Farah", "Gideon", "Helena",
    "Iris", "Jonas", "Keira", "Landon", "Mira", "Nolan", "Opal", "Priya",
    "Quinn", "Rhea", "Silas", "Talia", "Uma", "Vera", "Wade", "Xenia",
    "Yara", "Zane", "Amara", "Bennett", "Celine", "Devon", "Elodie", "Felix",
]
LAST_NAMES = [
    "Ashford", "Briar", "Caldwell", "Dunley", "Ellis", "Farrow", "Granger",
    "Hale", "Iverson", "Jensen", "Keller", "Lennox", "Marlow", "Norwood",
    "Ortega", "Parker", "Quincy", "Rowan", "Sterling", "Tanner", "Underwood",
    "Vale", "Winslow", "Yardley", "Zimmer", "Abbott", "Bexley", "Carver",
    "Drake", "Everly", "Frost", "Garnet",
]


@dataclass
class PhonebookExample:
    global_index: int
    sample_id: str
    target_position: int
    num_pairs: int
    target_name: str
    answer: str
    question: str
    conversation_text: str
    question_prompt: str
    full_prompt: str


def _name(index: int) -> str:
    first = FIRST_NAMES[index % len(FIRST_NAMES)]
    last = LAST_NAMES[(index // len(FIRST_NAMES)) % len(LAST_NAMES)]
    suffix = index // (len(FIRST_NAMES) * len(LAST_NAMES))
    return f"{first} {last} {suffix:03d}"


def _phone(seed: int, index: int) -> str:
    value = (seed * 1_000_003 + index * 97_531 + 10_000_000) % 10_000_000
    return f"{value:07d}"


def make_entries(num_pairs: int, seed: int) -> list[tuple[str, str]]:
    return [(_name(i), _phone(seed, i)) for i in range(num_pairs)]


def build_phonebook_text(entries: list[tuple[str, str]]) -> str:
    lines = [
        "PHONEBOOK",
        "Each line is: name | phone.",
        "",
    ]
    for name, phone in entries:
        lines.append(f"{name} | {phone}")
    return "\n".join(lines)


def build_question_prompt(name: str) -> str:
    return (
        f"What is the phone number of {name}?\n"
        "Answer with only the phone number. Do not explain."
    )


def build_full_prompt(phonebook_text: str, question_prompt: str) -> str:
    return f"{phonebook_text}\n\nQUESTION\n{question_prompt}"


def iter_examples(
    num_samples: int,
    num_pairs: int,
    seed: int = 42,
    position_mode: str = "uniform",
) -> Iterable[PhonebookExample]:
    rng = random.Random(seed)
    base_entries = make_entries(num_pairs, seed)
    for sample_idx in range(num_samples):
        entries = list(base_entries)
        rng.shuffle(entries)
        if position_mode == "uniform":
            target_position = rng.randrange(num_pairs)
        elif position_mode == "early":
            target_position = rng.randrange(max(1, num_pairs // 10))
        elif position_mode == "middle":
            lo = num_pairs * 45 // 100
            hi = max(lo + 1, num_pairs * 55 // 100)
            target_position = rng.randrange(lo, hi)
        elif position_mode == "late":
            lo = num_pairs * 9 // 10
            target_position = rng.randrange(lo, num_pairs)
        else:
            raise ValueError(f"Unknown position_mode: {position_mode}")
        target_name, answer = entries[target_position]
        phonebook_text = build_phonebook_text(entries)
        question_prompt = build_question_prompt(target_name)
        yield PhonebookExample(
            global_index=sample_idx,
            sample_id=f"phonebook-{sample_idx:06d}",
            target_position=target_position,
            num_pairs=num_pairs,
            target_name=target_name,
            answer=answer,
            question=question_prompt,
            conversation_text=phonebook_text,
            question_prompt=question_prompt,
            full_prompt=build_full_prompt(phonebook_text, question_prompt),
        )


def get_shard(
    examples: Iterable[PhonebookExample],
    shard_id: int,
    num_shards: int,
) -> list[PhonebookExample]:
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    return [ex for ex in examples if ex.global_index % num_shards == shard_id]


def apply_chat_template(tokenizer: Any, content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return f"{SYSTEM_PROMPT}\n\nUser:\n{content}\n\nAssistant:\n"
