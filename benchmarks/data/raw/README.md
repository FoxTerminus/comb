# Raw Benchmark Data Layout

Place official or downloaded benchmark files here before conversion.

Expected layout:

```text
benchmarks/data/raw/
  RULER/{dev,full}.jsonl
  SCBench/{dev,full}.jsonl
  LongBench/{dev,full}.jsonl
  LoCoMo/{dev,full}.jsonl
  LongCodeBench/{dev,full}.jsonl
```

`.json` files are also accepted when they contain either a list of rows or an object with a `data`, `examples`, or `instances` list.

The converters are deliberately offline: they do not download datasets and do not mutate raw files.

