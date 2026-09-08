import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

from datasets import Dataset
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.base as base


class ConcreteDataset(base.DatasetBase):
    name = "cache-test"

    def _init_data(self, split):
        raise NotImplementedError


class BucketCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.old_cache_dir = base.CACHE_DIR
        self.old_instances_per_file = base.NUM_INSTANCES_PER_FILE
        base.CACHE_DIR = self.temporary_directory.name
        base.NUM_INSTANCES_PER_FILE = 2

    def tearDown(self):
        base.CACHE_DIR = self.old_cache_dir
        base.NUM_INSTANCES_PER_FILE = self.old_instances_per_file
        self.temporary_directory.cleanup()

    @staticmethod
    def make_wrapped_dataset(rows=5):
        wrapped = object.__new__(ConcreteDataset)
        wrapped.model_name = "dummy-model"
        wrapped.max_input_length = 8
        wrapped.tokenizer = SimpleNamespace(pad_token_id=0)
        token_counts = [2, 3, 300, 301, 700][:rows]
        wrapped.data = Dataset.from_dict({
            "input_ids": [[10, 11]] * rows,
            "labels": [[12, 13]] * rows,
            "chunk_ids": [list(range(length)) for length in token_counts],
            "cross_attention_mask": [[1] * length for length in token_counts],
            "token_count": token_counts,
        })
        return wrapped

    def test_reuses_complete_cache_without_rewriting_files(self):
        wrapped = self.make_wrapped_dataset()
        first = wrapped.bucketing(local_rank=0, world_size=1)
        first_stats = [(path, os.stat(path).st_mtime_ns, os.path.getsize(path))
                       for _, path in first]

        second = wrapped.bucketing(local_rank=0, world_size=1)
        second_stats = [(path, os.stat(path).st_mtime_ns, os.path.getsize(path))
                        for _, path in second]

        self.assertEqual(first, second)
        self.assertEqual(first_stats, second_stats)
        self.assertTrue((Path(first[0][1]).parent / "manifest.json").is_file())

    def test_invalid_file_size_forces_atomic_rebuild(self):
        wrapped = self.make_wrapped_dataset()
        first = wrapped.bucketing(local_rank=0, world_size=1)
        damaged_path = first[0][1]
        original_size = os.path.getsize(damaged_path)
        with open(damaged_path, "ab") as handle:
            handle.write(b"damage")
        time.sleep(0.001)

        rebuilt = wrapped.bucketing(local_rank=0, world_size=1)

        self.assertEqual(first, rebuilt)
        self.assertEqual(os.path.getsize(damaged_path), original_size)
        self.assertFalse(list(Path(damaged_path).parent.glob("*.tmp-*")))

    def test_dataset_fingerprint_gets_separate_namespace(self):
        wrapped = self.make_wrapped_dataset()
        first = wrapped.bucketing(local_rank=0, world_size=1)
        wrapped.data = wrapped.data.select(range(4))
        second = wrapped.bucketing(local_rank=0, world_size=1)

        self.assertNotEqual(Path(first[0][1]).parent, Path(second[0][1]).parent)

    def test_payload_matches_original_bucketing_algorithm(self):
        wrapped = self.make_wrapped_dataset()
        actual_files = wrapped.bucketing(local_rank=0, world_size=1)
        actual = [
            (batch_size, self._normalized_records(pd.read_parquet(path)))
            for batch_size, path in actual_files
        ]

        frame = wrapped.data.to_pandas()
        frame["bucket"] = pd.cut(
            frame["token_count"],
            bins=base.BUCKET_SIZE,
            labels=range(len(base.BUCKET_SIZE) - 1),
            ordered=False,
        )
        expected = []
        for bucket, group in frame.groupby("bucket", observed=True):
            group = group.sample(frac=1, random_state=42)
            for start in range(0, len(group), base.NUM_INSTANCES_PER_FILE):
                shard = group.iloc[start:start + base.NUM_INSTANCES_PER_FILE]
                chunk_length = shard["token_count"].max()
                shard = shard.apply(
                    base.pad_tokens,
                    axis=1,
                    result_type="expand",
                    args=(chunk_length, wrapped.max_input_length,
                          "labels", wrapped.tokenizer.pad_token_id),
                )
                expected.append(
                    (base.BUCKET_BATCH_SIZE[bucket], self._normalized_records(shard))
                )

        self.assertEqual(actual, expected)

    @staticmethod
    def _normalized_records(frame):
        return [
            {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in row.items()
            }
            for row in frame.to_dict(orient="records")
        ]


if __name__ == "__main__":
    unittest.main()
