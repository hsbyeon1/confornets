"""Shared argparse helpers and device / test-case setup."""

import logging

import torch

from confornet.utils.dist import get_local_rank, get_world_size

logger = logging.getLogger(__name__)


def add_model_args(parser):
    """Model checkpoint + inference knobs shared by every script."""
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config-yaml", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-recycles", type=int, default=0)


def add_benchmark_args(parser):
    """Benchmark asset + sharding args for `run_*` scripts."""
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--assets-dir", type=str, default="./assets")
    parser.add_argument(
        "--test-case",
        type=str,
        default=None,
        help="Comma-separated test cases (default: all)",
    )
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-idx", type=int, default=0)


def add_confidence_args(parser):
    parser.add_argument(
        "--compute-confidence",
        action="store_true",
        help="Compute plddt/gpde/ptm/iptm and write per-sample CSV",
    )
    parser.add_argument(
        "--save-full-confidence",
        action="store_true",
        help="Also dump raw plddt/pde tensors (implies --compute-confidence)",
    )


def resolve_confidence_args(args):
    if args.save_full_confidence:
        args.compute_confidence = True


def setup_device():
    local_rank = get_local_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"Rank {local_rank}/{world_size}, device {device}")
    return local_rank, world_size, device
