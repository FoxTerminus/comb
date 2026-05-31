#!/usr/bin/env python3
"""LongBench runner for CombLlama and Hugging Face causal LMs.

The runner mirrors the project convention used for Comb evaluation:
``input_ids`` is the question/prompt, ``chunk_ids`` is the context, generation
is single-sample greedy decoding, and results are written under
``LongBench/results``.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comb.models.CombLlama import CombLlamaConfig, CombLlamaForConditionalGeneration  # noqa: E402
from comb.models.tp_checkpoint import merge_state_dict_from_tp  # noqa: E402


QA_DATASETS = {"hotpotqa", "2wikimqa", "musique"}
SUMMARY_DATASETS = {"multi_news", "samsum"}
DEFAULT_DATASETS = ["hotpotqa", "2wikimqa", "musique", "multi_news", "samsum"]

INSTRUCTIONS = {
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: "
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: "
    ),
    "musique": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\nQuestion: "
    ),
    "multi_news": (
        "You are an AI assistant. "
        "Read the provided text and produce a concise summary. "
        "Capture the main points without unnecessary details."
    ),
    "samsum": (
        "You are an AI assistant. "
        "Read the provided text and produce a concise summary. "
        "Capture the main points without unnecessary details."
    ),
}

CORRECT_ANSWERS = {
    "hotpotqa": {
        "Which mountain is higher, Tongshanjiabu or Himalchuli?": [
            "Himalchuli",
            "Himalchuli is higher than Tongshanjiabu.",
        ],
        "Band-e-Amir Dragons is named after the lakes in which Afghan national park?": [
            "Band-e-Amir National Park"
        ],
        "In what event was Harold Davis a former record holder, but now is held by Usain Bolt?": [
            "100 metres",
            "100 m",
        ],
        "What career led Brandon James Routh to move to the city where Brian Ralston lives?": [
            "an acting career",
            "acting",
        ],
        "What is the character of fictional character Claire Fraser in a British-American television drama series developed by Ronald D. Moore ?": [
            "Claire is a married World War II nurse",
            "A nurse.",
        ],
        "Which composer was wrote his music most recently, Michael Tippett or Luigi Cherubini?": [
            "Michael Tippett",
            "Michael Kemp Tippett",
        ],
        "Which filmmaker was known for animation, Lev Yilmaz or Pamela B. Green?": [
            "Lev Yilmaz",
            "Lev Yilmaz was known for animation.",
        ],
    },
    "musique": {
        "When did the party who gained control of congress in the midterm elections in 1946 take control of the determiner of rules of the US House and US Senate?": [
            "January 3, 1947",
            "January, 1947",
        ],
        "What pantheon is the God of the underworld in ancient Egypt a part of?": [
            "Egyptian pantheon",
            "The God of the underworld is a part of the Egyptian pantheon.",
        ],
        "What other recognition did the Oscar winner for Best Actor in 2006 receive?": [
            "nominated for an Academy Award for Best Supporting Actor",
            "Academy Award for Best Supporting Actor",
            "Best Supporting Actor",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LongBench for CombLlama or HF models.")
    parser.add_argument("--model-type", choices=["comb", "hf"], default="comb")
    parser.add_argument("--checkpoint", type=str, default=None, help="Comb checkpoint dir/file.")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--limit", type=int, default=0, help="Per-dataset testcase cap; 0 means full.")
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--qa-max-new-tokens", type=int, default=128)
    parser.add_argument("--summary-max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(prediction_tokens))
    recall = num_same / max(1, len(ground_truth_tokens))
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    try:
        from rouge import Rouge

        return float(Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"])
    except Exception:
        return 0.0


def score_prediction(dataset_name: str, prediction: str, answers: list[str]) -> float:
    if not answers:
        return 0.0
    if dataset_name in QA_DATASETS:
        return max(f1_score(prediction, answer) for answer in answers)
    return max(rouge_l_score(prediction, answer) for answer in answers)


def clean_prediction(text: str) -> str:
    text = text.replace("<|eot_id|>", "").replace("<|end_of_text|>", "")
    text = re.sub(r"^\s*assistant\s*\n*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def dataset_max_new_tokens(dataset_name: str, args: argparse.Namespace) -> int:
    return args.qa_max_new_tokens if dataset_name in QA_DATASETS else args.summary_max_new_tokens


def build_query(dataset_name: str, example: dict[str, Any]) -> str:
    instruction = INSTRUCTIONS[dataset_name]
    if dataset_name not in DEFAULT_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if dataset_name in SUMMARY_DATASETS:
        task_input = example["input"]
        return instruction + ("\n\n" + task_input if task_input else "")
    return instruction + example["input"]


def build_hf_prompt(dataset_name: str, example: dict[str, Any]) -> str:
    context = example["context"]
    query = build_query(dataset_name, example)
    return context.rstrip() + "\n\n" + query


def chat_ids(tokenizer, content: str, add_generation_prompt: bool = False) -> list[int]:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )


def prepare_answers(dataset_name: str, example: dict[str, Any]) -> list[str]:
    corrected = CORRECT_ANSWERS.get(dataset_name, {}).get(example.get("input", ""))
    if corrected is not None:
        return corrected
    answers = example.get("answers", [])
    if isinstance(answers, str):
        return [answers]
    return list(answers)


def load_longbench_dataset(dataset_name: str, args: argparse.Namespace):
    return load_dataset(
        "THUDM/LongBench",
        dataset_name,
        split="test",
        trust_remote_code=True,
        download_mode="reuse_dataset_if_exists",
    )


def dtype_from_arg(arg: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[arg]


def load_comb_model(args: argparse.Namespace):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --model-type comb.")
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        rank_files = sorted(checkpoint.glob("rank_*.pt"))
        if not rank_files:
            raise FileNotFoundError(f"No rank_*.pt files found in {checkpoint}")
        shards = []
        config_dict = None
        step = None
        for path in rank_files:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            shards.append(ckpt["model"])
            config_dict = ckpt.get("config", config_dict)
            step = ckpt.get("step", step)
        state = merge_state_dict_from_tp(shards) if len(shards) > 1 else shards[0]
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        config_dict = ckpt.get("config") if isinstance(ckpt, dict) else None
        step = ckpt.get("step") if isinstance(ckpt, dict) else None
    if config_dict is None:
        raise ValueError("Comb checkpoint does not contain `config`.")
    config = CombLlamaConfig(**config_dict)
    model = CombLlamaForConditionalGeneration(config)
    model.load_state_dict(state, strict=True)
    model.to(args.device, dtype=dtype_from_arg(args.dtype))
    model.eval()
    return model, step


def load_hf_model(args: argparse.Namespace):
    dtype = dtype_from_arg(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    return model


def generate_comb(model, tokenizer, dataset_name: str, example: dict[str, Any], args: argparse.Namespace) -> str:
    question = build_query(dataset_name, example)
    context = example["context"]
    input_ids = torch.tensor(chat_ids(tokenizer, question, add_generation_prompt=True), device=args.device).unsqueeze(0)
    chunk_ids = torch.tensor(tokenizer(context, add_special_tokens=True)["input_ids"], device=args.device).unsqueeze(0)
    max_new_tokens = dataset_max_new_tokens(dataset_name, args)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            chunk_ids=chunk_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=0.0,
        )
    generated_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    return clean_prediction(tokenizer.decode(generated_ids, skip_special_tokens=True))


def generate_hf(model, tokenizer, dataset_name: str, example: dict[str, Any], args: argparse.Namespace) -> str:
    prompt = build_hf_prompt(dataset_name, example)
    input_ids = torch.tensor(chat_ids(tokenizer, prompt, add_generation_prompt=True), device=model.device).unsqueeze(0)
    max_new_tokens = dataset_max_new_tokens(dataset_name, args)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    return clean_prediction(tokenizer.decode(generated_ids, skip_special_tokens=True))


def run_dataset(model, tokenizer, dataset_name: str, args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    dataset = load_longbench_dataset(dataset_name, args)
    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    scores = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(total), desc=dataset_name):
            example = dataset[idx]
            answers = prepare_answers(dataset_name, example)
            if args.model_type == "comb":
                prediction = generate_comb(model, tokenizer, dataset_name, example, args)
            else:
                prediction = generate_hf(model, tokenizer, dataset_name, example, args)
            score = score_prediction(dataset_name, prediction, answers)
            scores.append(score)
            row = {
                "dataset": dataset_name,
                "index": idx,
                "prediction": prediction,
                "answers": answers,
                "score": score,
                "metric": "qa_f1" if dataset_name in QA_DATASETS else "rouge_l",
                "input": example.get("input", ""),
                "context_length_chars": len(example.get("context", "")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    elapsed = time.time() - start
    mean_score = sum(scores) / max(1, len(scores))
    return {
        "dataset": dataset_name,
        "metric": "qa_f1" if dataset_name in QA_DATASETS else "rouge_l",
        "score": mean_score,
        "num_examples": len(scores),
        "elapsed_sec": elapsed,
        "output": str(output_path),
    }


def main() -> None:
    args = parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DEFAULT_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported LongBench datasets: {unknown}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "results"
    prefix = args.output_prefix

    tokenizer_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == "comb":
        model, step = load_comb_model(args)
        if prefix is None:
            prefix = f"step_{step}" if step is not None else "comb"
    else:
        model = load_hf_model(args)
        if prefix is None:
            prefix = args.model_name.rstrip("/").split("/")[-1].lower().replace(".", "").replace("-", "_")

    summaries = []
    for dataset_name in datasets:
        suffix = "full" if args.limit <= 0 else str(args.limit)
        output_path = output_dir / f"{prefix}_{dataset_name}_{suffix}.jsonl"
        summary = run_dataset(model, tokenizer, dataset_name, args, output_path)
        summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    all_summary_path = output_dir / f"{prefix}_summary.json"
    all_summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
