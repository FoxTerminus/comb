"""LoCoMo data loading and prompt construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config import DEFAULT_DATA_FILE


CONV_START_PROMPT = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

QA_PROMPT = """
Based on the above conversations, answer the following question.
Answer with only the short answer.
Use exact words from the conversations whenever possible.
Use at most 8 words.
Do not explain.
Do not repeat the question.

Question: {question}
"""

QA_PROMPT_CAT_5 = """
Based on the above conversations, answer the following question.
Answer with only the short answer.
If the answer is not available in the conversations, answer exactly: No information available.
Do not explain.
Do not repeat the question.

Question: {question}
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant whose job is to understand a long "
    "multi-day conversation and answer questions based only on that conversation. "
    "Return concise short answers only."
)


@dataclass
class LocomoExample:
    global_index: int
    sample_id: str
    qa_index: int
    category: int
    question: str
    answer: str
    evidence: list[str]
    conversation_text: str
    question_prompt: str
    full_prompt: str


def ensure_data_file(data_file: str | Path | None = None) -> Path:
    """Return a LoCoMo data path, downloading the cached HF file if needed."""
    path = Path(data_file) if data_file else DEFAULT_DATA_FILE
    if path.exists():
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"LoCoMo data file not found at {path}. Install huggingface_hub "
            "or pass --data-file explicitly."
        ) from exc

    downloaded = hf_hub_download(
        "Percena/locomo-mc10",
        filename="raw/locomo10.json",
        repo_type="dataset",
    )
    return Path(downloaded)


def load_samples(data_file: str | Path | None = None) -> list[dict[str, Any]]:
    with ensure_data_file(data_file).open("r", encoding="utf-8") as f:
        return json.load(f)


def session_numbers(conversation: dict[str, Any]) -> list[int]:
    nums = []
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("_date_time") and value:
            nums.append(int(key.split("_")[-1]))
    return sorted(nums)


def build_conversation_text(conversation: dict[str, Any]) -> str:
    speaker_a = conversation.get("speaker_a", "speaker A")
    speaker_b = conversation.get("speaker_b", "speaker B")
    text = CONV_START_PROMPT.format(speaker_a=speaker_a, speaker_b=speaker_b)
    for sess_num in session_numbers(conversation):
        date = conversation.get(f"session_{sess_num}_date_time", "")
        text += f"DATE: {date}\nCONVERSATION:\n"
        for dialog in conversation.get(f"session_{sess_num}", []):
            speaker = dialog.get("speaker", "Unknown")
            utterance = dialog.get("text") or dialog.get("clean_text") or ""
            text += f'{speaker} said, "{utterance}"\n'
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text += f'and shared {dialog["blip_caption"]}.\n'
        text += "\n"
    return text.strip()


def build_question_prompt(
    question: str,
    category: int,
    adversarial_answer: str | None = None,
) -> str:
    if category == 2:
        question = question + " Use DATE of CONVERSATION to answer with an approximate date."
    if category == 5:
        if adversarial_answer:
            question = (
                f"{question} Select the correct answer: "
                f"(a) {adversarial_answer} (b) Not mentioned in the conversation."
            )
        return QA_PROMPT_CAT_5.format(question=question).strip()
    return QA_PROMPT.format(question=question).strip()


def build_full_prompt(conversation_text: str, question_prompt: str) -> str:
    return f"{conversation_text}\n\n{question_prompt}"


def iter_examples(data_file: str | Path | None = None) -> Iterable[LocomoExample]:
    global_index = 0
    for sample in load_samples(data_file):
        conversation_text = build_conversation_text(sample["conversation"])
        for qa_index, qa in enumerate(sample["qa"]):
            question = str(qa["question"])
            category = int(qa["category"])
            adversarial_answer = qa.get("adversarial_answer")
            question_prompt = build_question_prompt(question, category, adversarial_answer)
            answer = qa.get("answer", "No information available")
            yield LocomoExample(
                global_index=global_index,
                sample_id=str(sample["sample_id"]),
                qa_index=qa_index,
                category=category,
                question=question,
                answer=str(answer),
                evidence=list(qa.get("evidence", [])),
                conversation_text=conversation_text,
                question_prompt=question_prompt,
                full_prompt=build_full_prompt(conversation_text, question_prompt),
            )
            global_index += 1


def get_shard(
    examples: Iterable[LocomoExample],
    shard_id: int,
    num_shards: int,
) -> list[LocomoExample]:
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
