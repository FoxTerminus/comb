import tempfile

import torch
import torch.distributed as dist
from torch import nn

from baselines.SambaY.models.SambaY_megatron import GatedPairColumnParallelLinear, apply_tensor_parallelism
from baselines.SambaY.tests.test_sambay_smoke import build_tiny_model


def _init_single_rank_group():
    if dist.is_initialized():
        return dist.group.WORLD
    handle = tempfile.NamedTemporaryFile(delete=True)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{handle.name}",
        rank=0,
        world_size=1,
    )
    return dist.group.WORLD


def test_gated_pair_column_parallel_linear_preserves_gate_value_layout():
    group = _init_single_rank_group()
    linear = nn.Linear(4, 12, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(48, dtype=torch.float32).view(12, 4))
    tp_linear = GatedPairColumnParallelLinear.from_linear(linear, group)
    x = torch.randn(2, 3, 4)

    actual_gate, actual_value = tp_linear(x).chunk(2, dim=-1)
    expected_gate, expected_value = linear(x).chunk(2, dim=-1)

    torch.testing.assert_close(actual_gate, expected_gate)
    torch.testing.assert_close(actual_value, expected_value)


def test_sambay_tp_adapter_accepts_world_size_one():
    group = _init_single_rank_group()
    model = build_tiny_model()

    apply_tensor_parallelism(model, group)

    input_ids = torch.tensor([[1, 5, 6, 7]])
    shift_labels = torch.tensor([[-100, 5, 6, 7]])
    cu_seqlens_q = torch.tensor([0, input_ids.shape[1]], dtype=torch.int32)
    outputs = model(
        input_ids=input_ids,
        shift_labels=shift_labels,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=input_ids.shape[1],
        use_cache=False,
    )

    assert torch.isfinite(outputs.loss)
