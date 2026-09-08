#!/usr/bin/env python3
"""Serve a locally trained Comb checkpoint through the official HTTP API."""

from __future__ import annotations

import argparse
import asyncio

from comb.entrypoints.api_server import run_server
from comb.supported_models import COMB_MODEL_MAPPING


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--comb-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--pic-memory-utilization", type=float, default=0.50)
    parser.add_argument("--pbc-memory-utilization", type=float, default=0.30)
    parser.add_argument("--disable-log-stats", action="store_true")
    args = parser.parse_args()
    COMB_MODEL_MAPPING[args.base_model] = args.comb_model
    # ``run_server`` expects the same attribute names as the official CLI.
    args.model = args.base_model
    args.num_instances = 1
    args.pic_separated = False
    args.root_path = ""
    args.log_level = "info"
    asyncio.run(run_server(args))


if __name__ == "__main__":
    main()
