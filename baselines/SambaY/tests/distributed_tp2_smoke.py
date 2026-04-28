"""Two-rank CPU smoke for SambaY tensor parallel forward/backward.

Run with:
    python -m torch.distributed.run --nproc_per_node=2 baselines/SambaY/tests/distributed_tp2_smoke.py
"""

import torch
import torch.distributed as dist

from baselines.SambaY.models.SambaY_megatron import apply_tensor_parallelism
from baselines.SambaY.tests.test_sambay_smoke import build_tiny_model


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.manual_seed(1234)
    model = build_tiny_model()
    apply_tensor_parallelism(model, dist.group.WORLD)
    model.train()

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
    if not torch.isfinite(outputs.loss):
        raise RuntimeError(f"Rank {rank} produced non-finite loss: {outputs.loss}")
    outputs.loss.backward()
    loss = outputs.loss.detach().clone()
    losses = [torch.empty_like(loss) for _ in range(dist.get_world_size())]
    dist.all_gather(losses, loss)
    if not torch.allclose(torch.stack(losses), losses[0].expand_as(torch.stack(losses)), atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"Rank losses diverged: {[float(item.item()) for item in losses]}")
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"tp2_loss_avg={float(loss.item() / dist.get_world_size()):.6f}")

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
    if generated.shape != (1, input_ids.shape[1] + 1):
        raise RuntimeError(f"Rank {rank} generated unexpected shape: {tuple(generated.shape)}")
    if rank == 0:
        print(f"tp2_generate_shape={tuple(generated.shape)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
