"""Distributed utilities."""

import os


def get_local_rank() -> int:
    """Get local rank from environment variables (torchrun)."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "FLUX_TASK_RANK" in os.environ and "NRANKS_PER_NODE" in os.environ:
        return int(os.environ["FLUX_TASK_RANK"]) % int(os.environ["NRANKS_PER_NODE"])
    else:
        return 0


def get_world_size() -> int:
    """Get world size from environment variables (torchrun)."""
    return int(os.environ.get("WORLD_SIZE", 1))
