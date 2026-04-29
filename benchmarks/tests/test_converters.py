from benchmarks.scripts.converters.synthetic import build_synthetic_examples
from benchmarks.scripts.converters.official import (
    convert_locomo,
    convert_longbench,
    convert_longcodebench,
    convert_ruler,
    convert_scbench,
)
from benchmarks.scripts.converters.convert_raw import convert_file
from benchmarks.scripts.utils.io import read_jsonl
from benchmarks.scripts.schema import BenchmarkExample


def test_synthetic_smoke_has_all_benchmarks():
    examples = [BenchmarkExample.from_dict(row.to_dict()) for row in build_synthetic_examples("smoke")]
    assert {example.benchmark for example in examples} == {
        "RULER",
        "SCBench",
        "LongBench",
        "LoCoMo",
        "LongCodeBench",
    }


def test_synthetic_dev_has_more_than_smoke():
    assert len(build_synthetic_examples("dev")) > len(build_synthetic_examples("smoke"))


def test_ruler_official_converter_preserves_task_metadata():
    rows = [
        {
            "id": "r1",
            "task": "niah_single_1",
            "context": "needle is blue",
            "question": "What color is the needle?",
            "answer": "blue",
            "task_type": "retrieval",
            "needle_position": "middle",
            "context_length": 4096,
        }
    ]
    example = convert_ruler(rows, "dev")[0]
    assert BenchmarkExample.from_dict(example.to_dict()).benchmark == "RULER"
    assert example.metadata["task_type"] == "retrieval"
    assert example.metadata["needle_position"] == "middle"
    assert example.metadata["target_context_length"] == 4096


def test_scbench_official_converter_preserves_cache_metadata():
    rows = [
        {
            "id": "s1",
            "task": "shared_kv",
            "shared_context": "alpha maps to beta",
            "query": "What does alpha map to?",
            "answer": "beta",
            "shared_context_id": "ctx-1",
            "query_index": 2,
        }
    ]
    example = convert_scbench(rows, "dev")[0]
    assert example.metadata["shared_context_id"] == "ctx-1"
    assert example.metadata["query_index"] == 2
    assert example.metadata["cache_reuse"] is True


def test_longbench_official_converter_maps_task_metric():
    rows = [
        {
            "id": "l1",
            "dataset": "qasper",
            "context": "Mira leads the lab.",
            "question": "Who leads the lab?",
            "answers": ["Mira"],
            "language": "en",
        }
    ]
    example = convert_longbench(rows, "dev")[0]
    assert example.task == "qasper"
    assert example.metadata["official_task"] == "qasper"
    assert example.metadata["metric"] == "f1"


def test_locomo_official_converter_renders_dialogue():
    rows = [
        {
            "id": "m1",
            "conversation": [
                {"speaker": "Dana", "text": "Dinner is at Cedar Hall."},
                {"speaker": "Kai", "text": "Noted."},
            ],
            "question": "Where is dinner?",
            "answer": "Cedar Hall",
            "question_type": "single_hop",
            "temporal": False,
        }
    ]
    example = convert_locomo(rows, "dev")[0]
    assert "Dana: Dinner is at Cedar Hall." in example.context
    assert example.metadata["question_type"] == "single_hop"


def test_longcodebench_official_converter_preserves_repo_metadata():
    rows = [
        {
            "id": "c1",
            "repo_context": "def normalize_name(x): return x.strip().lower()",
            "instruction": "Which function normalizes names?",
            "answer": "normalize_name",
            "repo": "demo",
            "file": "utils.py",
            "language": "python",
        }
    ]
    example = convert_longcodebench(rows, "dev")[0]
    assert example.benchmark == "LongCodeBench"
    assert example.metadata["repo"] == "demo"
    assert example.metadata["language"] == "python"


def test_convert_file_writes_only_requested_output_root(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        '{"id":"r-cli","task":"niah","context":"the key is amber",'
        '"question":"What is the key?","answer":"amber"}\n',
        encoding="utf-8",
    )
    output_root = tmp_path / "processed"
    examples = convert_file("RULER", "dev", raw_path, output_root)
    assert len(examples) == 1
    rows = read_jsonl(output_root / "dev" / "RULER.jsonl")
    assert rows[0]["id"] == "r-cli"
