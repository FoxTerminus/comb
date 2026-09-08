<p align="center">
  <picture>
    <img alt="COMB" src="assets/logo.svg" width=55%>
  </picture>
</p>

# COMB
COMB is a plug-and-play caching system for long-context LLM serving.

## Local Frozen32 development

This checkout also contains the Frozen32 training and evaluation work. Start with
the [project guide](docs/project-layout.md), [training guide](training/README.md),
and [test instructions](tests/README.md). Shell launchers live under
[`scripts/`](scripts/README.md); historical reproduction notes live under
[`docs/archive/`](docs/archive/2026-08-25-reproduction-status.md).

The serving examples below describe the original COMB implementation. The
Frozen32 HF model and its checkpoints are a separate model path; they are not
interchangeable with the original COMB/vLLM checkpoints.

## Code Structure
```
COMB
├── assets
├── benchmarks                   # For benchmarking
├── comb
│   ├── entrypoints
│   │   ├── api_server.py        # For online server
│   │   └── comb.py              # For offline inference
│   ├── integration
│   │   ├── hf                   # hf transformers backend
│   │   ├── vllm                 # vLLM backend
│   │   └── __init__.py
│   ├── storage
│   │   ├── chunk_processor.py   # For generating PIC
│   │   ├── pic_allocator.py     # For allocating memory
│   │   ├── pic_manager.py       # For managing PIC
│   │   └── pic_utils.py
│   ├── transfer
│   │   └── cuda_ipc_utils.py    # For inter-process communication
│   ├── __init__.py
│   ├── output.py
│   └── supported_models.py
├── data
├── examples                     # For use case
├── training                     # Training modules and DeepSpeed configs
├── scripts                      # Shell launchers (training / benchmarks)
├── tests                        # CPU tests (training / benchmarks)
├── docs                         # Project guide and historical reports
├── pytest.ini                   # Unified test discovery
├── environment.yml
└── requirements.txt
```

## Getting Started
Run the following commands to prepare the environment. We recommend appending two `export` commands to the end of `~/.bashrc`.
```bash
export PYTHONPATH=~/Comb:$PYTHONPATH
export TOKENIZERS_PARALLELISM=true
pip install -r requirements.txt
```

Install vllm. (Recommended for efficiency and benchmarking)
```bash
pip install vllm
```

Currently we only support `meta-llama/Llama-3.1-8B-Instruct` and `deepseek-ai/DeepSeek-V2-Lite-Chat`. If you want to use another model, you can also train a Comb model by yourself through following our [instructions](training/README.md).

## Usage

You can find examples in the folder `examples`.

- [basic.py](examples/basic.py) for offline inference.
- [online_serving.py](examples/online_serving.py) for server.

## Benchmark

See [Instructions](benchmarks/README.md).

## Demo

In this example, we simulate two requests with different prefixes. The requests contain the same question and retrieved context, enabling the KV cache to be reused through PIC.

<p align="center">
  <h3 align="center">🖥️ Demo</h3>
  <video src="https://github.com/user-attachments/assets/ce6bf940-1e54-4b4c-9ea6-c2ae1afe1679"
         controls
         muted
         playsinline>
    <a href="https://github.com/user-attachments/assets/ce6bf940-1e54-4b4c-9ea6-c2ae1afe1679">Demo of TTFT speedup using COMB.</a>
  </video>
</p>
