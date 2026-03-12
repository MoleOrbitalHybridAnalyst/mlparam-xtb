from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from mace_jax.tools import bundle as bundle_tools
from mace_jax.cli.mace_jax_from_torch import convert_model
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.nnx_utils import state_to_pure_dict
from mace_jax.data.utils import AtomicNumberTable
from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.gto import make_basis_array
from pyscfad.ml.xtb.param import make_param_array

from data import QMMMDataset, DataLoader
from train_helper import XTBModel, scalar_node_feature_indices


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test XTBModel with real data/model")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/chhli/projects/Enzyme_Kinetics_OPES_flooding/chorismate_mutase/"
            "qmmm_single_point/wb97x-3c_refined_lno/no_constraints_qm3.1/neb_data.npz"
        ),
        help="NPZ file with QM/MM data",
    )
    p.add_argument(
        "--train_list",
        type=Path,
        default=Path("torch_example/neb.train.list.0"),
        help="Txt file of frame indices to load",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/chhli/projects/Enzyme_Kinetics_OPES_flooding/chorismate_mutase/"
            "ml/train/mace/train3d/bs5_lr3e-3_wt0.5_wv0.5.best.model"
        ),
        help="MACE-JAX bundle dir or ckpt (config.json+params.msgpack). "
        "Override this; the default path from torch_example/train.sh is a torch checkpoint and will not load.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for loader",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup calls to trigger compilation",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of timed runs after warmup",
    )
    return p.parse_args()


def main():
    args = parse_args()

    missing = [p for p in [args.data, args.train_list, args.model] if not p.exists()]
    if missing:
        print("Skipping: missing paths:")
        for m in missing:
            print(" -", m)
        return

    def _load_mace_module(model_path: Path):
        # Try native JAX bundle first
        try:
            bundle = bundle_tools.load_model_bundle(str(model_path), dtype="float64")
            mace_module = nnx.merge(bundle.graphdef, bundle.params)
            z_table = AtomicNumberTable([int(z) for z in mace_module.atomic_numbers])
            cutoff = float(mace_module.r_max)
            return mace_module, z_table, cutoff
        except Exception:
            pass

        # Fallback: Torch checkpoint -> JAX via convert_model
        try:
            import torch  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "Torch not available to convert torch checkpoint; "
                "provide a JAX bundle instead."
            ) from exc

        torch_model = torch.load(model_path, map_location="cpu")
        if isinstance(torch_model, dict) and "model" in torch_model:
            torch_model = torch_model["model"]
        torch_model.eval()

        config = extract_config_mace_model(torch_model)
        if "error" in config:
            raise RuntimeError(config["error"])
        graphdef, state, _ = convert_model(torch_model, config)
        mace_module = nnx.merge(graphdef, state)
        z_table = AtomicNumberTable([int(z) for z in mace_module.atomic_numbers])
        cutoff = float(mace_module.r_max)
        return mace_module, z_table, cutoff

    try:
        mace_module, z_table, cutoff = _load_mace_module(args.model)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Failed to load/convert model at {args.model}: {exc}")
        return

    indices = np.loadtxt(args.train_list, dtype=int)
    dataset = QMMMDataset(
        npz_files=[str(args.data)],
        dataslices=[indices],
        z_table=z_table,
        cutoff=cutoff,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    batch = jax.device_put(batch)

    # precompute static max sizes for padding
    max_qm = max(s.z.shape[0] for s in dataset.samples)
    max_mm = max(s.z_mm.shape[0] for s in dataset.samples)

    max_z = max(s.z.max() for s in dataset.samples)

    basis = make_basis_array(xtb_basis.get_basis_filename(), max_number=max_z)
    param = make_param_array(basis, max_number=max_z)
    param.dipgam = jnp.array(param.gam)
    param.quadgam = jnp.array(param.gam)


    rngs = nnx.Rngs(params=jax.random.key(0))
    # determine node feature dim by a single MACE forward pass (no state context needed)
    mace_out = mace_module(batch, compute_node_feats=True)
    scalar_indices = scalar_node_feature_indices(mace_module)
    node_feat_dim = (
        int(scalar_indices.shape[0])
        if scalar_indices is not None
        else int(mace_out["node_feats"].shape[-1])
    )

    model = XTBModel(
        mace_model=mace_module,
        xtb_param=param,
        basis=basis,
        rngs=rngs,
        node_feat_dim=node_feat_dim,
        max_qm=max_qm,
        max_mm=max_mm,
    )

    model = nnx.jit(model)
    for _ in range(max(args.warmup, 0)):
        energy = model(batch)
        jax.block_until_ready(energy)
    for _ in range(max(args.repeat, 0)):
        energy = model(batch)
        jax.block_until_ready(energy)
    print("Energies (eV):", energy)
    if "energy" in batch:
        print("Reference (eV):", batch["energy"])
        print("MAE (eV):", jnp.abs(energy - batch["energy"]).mean())


if __name__ == "__main__":
    main()
