from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

from comb.entrypoints import api_server


class FakeCOMB:
    def __init__(self, **_kwargs) -> None:
        self.state_lock = threading.Lock()
        self.active = 0
        self.maximum_active = 0
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def generate_for_single_request(self, prompt, **_kwargs):
        with self.state_lock:
            self.active += 1
            self.calls += 1
            self.maximum_active = max(self.maximum_active, self.active)
        self.started.set()
        self.release.wait(timeout=5)
        time.sleep(0.01)
        with self.state_lock:
            self.active -= 1
        return prompt["value"]


class AsyncCOMBConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def make_engine(self):
        with patch.object(api_server, "COMB", FakeCOMB):
            return api_server.AsyncCOMB(model="test")

    async def test_overlapping_requests_use_one_sync_worker(self) -> None:
        engine = self.make_engine()
        try:
            tasks = [
                asyncio.create_task(engine.generate({"value": value}))
                for value in range(12)
            ]
            await asyncio.sleep(0.5)
            self.assertTrue(all(task.done() for task in tasks))
            values = [task.result() for task in tasks]
            self.assertEqual(values, list(range(12)))
            self.assertEqual(engine.comb.calls, 12)
            self.assertEqual(engine.comb.maximum_active, 1)
        finally:
            engine._executor.shutdown(wait=True)

    async def test_cancelled_waiter_does_not_release_worker_serialization(self) -> None:
        engine = self.make_engine()
        engine.comb.release.clear()
        try:
            first = asyncio.create_task(engine.generate({"value": 1}))
            deadline = asyncio.get_running_loop().time() + 2
            while not engine.comb.started.is_set():
                self.assertLess(asyncio.get_running_loop().time(), deadline)
                await asyncio.sleep(0.01)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first

            second = asyncio.create_task(engine.generate({"value": 2}))
            await asyncio.sleep(0.05)
            self.assertEqual(engine.comb.calls, 1)
            self.assertEqual(engine.comb.maximum_active, 1)

            engine.comb.release.set()
            await asyncio.sleep(0.1)
            self.assertTrue(second.done())
            self.assertEqual(second.result(), 2)
            self.assertEqual(engine.comb.calls, 2)
            self.assertEqual(engine.comb.maximum_active, 1)
        finally:
            engine.comb.release.set()
            engine._executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
