import torch
from openfold3.projects.of3_all_atom.model import OpenFold3
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate, OF3ProjectEntry

DEFAULT_MODEL_UPDATE = ModelUpdate(
    custom={
        "settings": {
            "blocks_per_ckpt": 1,
            "ckpt_intermediate_steps": True,
            "memory": {
                "eval": {
                    "chunk_size": 4,
                    "offload_inference": {
                        "confidence_heads": True,
                        "token_cutoff": 2800,
                    },
                }
            },
        },
        "architecture": {"heads": {"pae": {"enabled": True}}},
    }
)


def load_model(
    checkpoint_path: str,
    model_update: ModelUpdate | None = None,
    device: torch.device | None = None,
    subsampled_all_msa_rows: int | None = None,
) -> OpenFold3:
    model_update = model_update or DEFAULT_MODEL_UPDATE
    model_cfg = OF3ProjectEntry().get_model_config_with_update(model_update)

    if subsampled_all_msa_rows is not None:
        if subsampled_all_msa_rows <= 0:
            raise ValueError("subsampled_all_msa_rows must be > 0")
        msa_embedder_cfg = model_cfg.architecture.msa.msa_module_embedder
        msa_embedder_cfg.subsample_all_msa = True
        msa_embedder_cfg.subsample_main_msa = False
        msa_embedder_cfg.min_subsampled_all_msa = subsampled_all_msa_rows
        msa_embedder_cfg.max_subsampled_all_msa = subsampled_all_msa_rows

    model = OpenFold3(model_cfg)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # OF3p2 checkpoints (openfold3 >= 0.4.0) omit the `version_tensor` buffer.
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing in ([], ["version_tensor"]) and unexpected == [], (
        f"Unexpected load_state_dict mismatch: missing={missing}, unexpected={unexpected}"
    )

    if device:
        model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model
