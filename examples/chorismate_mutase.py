from __future__ import annotations

import argparse
import time
import optax
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
        default=500,
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

    # Scan dataset for energy and forces
    energies = []
    forces = []
    for s in dataset.samples:
        if s.energy is not None:
            energies.append(s.energy)
        if s.forces is not None:
            forces.append(s.forces)
        if s.forces_mm is not None:
            forces.append(s.forces_mm)
            
    energies = np.concatenate([np.atleast_1d(e) for e in energies])
    forces = np.concatenate([np.atleast_2d(f) for f in forces], axis=0)
    
    e_scale = float(1.0 / max(np.ptp(energies) ** 2, 1e-8))
    f_scale = float(1.0 / max(np.ptp(forces) ** 2, 1e-8))
    print(f"e_scale: {e_scale:.4e}, f_scale: {f_scale:.4e}")

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

    base_lr = 1e-5
    warmup_lr = 1e-7
    warmup_steps = max(args.warmup, 0)

    if warmup_steps > 0:
        lr_schedule = optax.linear_schedule(
            init_value=warmup_lr,
            end_value=base_lr,
            transition_steps=warmup_steps
        )
    else:
        lr_schedule = base_lr

    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule), wrt=nnx.Param)

    graphdef, state = nnx.split((model, optimizer))

    @jax.jit
    def train_step(state, batch, e_scale, f_scale):
        model, optimizer = nnx.merge(graphdef, state)
        def loss_fn(model):
            def energy_fn(positions, positions_mm):
                batch_local = dict(batch)
                batch_local["positions"] = positions
                batch_local["positions_mm"] = positions_mm
                e_pred = model(batch_local)
                if "graph_mask" in batch_local:
                    e_pred = jnp.where(batch_local["graph_mask"], e_pred, 0.0)
                return jnp.sum(e_pred), e_pred

            (sum_e, e_pred), (fqm, fmm) = nnx.value_and_grad(
                energy_fn, argnums=(0, 1), has_aux=True
            )(batch["positions"], batch["positions_mm"])
            fqm = -fqm
            fmm = -fmm

            e_diff = e_pred - batch["energy"]
            fqm_diff = fqm - batch["forces"]

            if "graph_mask" in batch:
                g_mask = batch["graph_mask"].astype(e_diff.dtype)
                n_mask = batch["node_mask"].astype(fqm_diff.dtype)[..., None]
                num_graphs = jnp.maximum(jnp.sum(g_mask), 1.0)
                e_offset = jnp.sum(e_diff * g_mask) / num_graphs
                loss_value_e = jnp.sum(((e_diff - e_offset) ** 2) * g_mask) / num_graphs * e_scale
                num_qm_components = jnp.maximum(jnp.sum(n_mask) * 3.0, 1.0)
                loss_value_f = jnp.sum((fqm_diff ** 2) * n_mask) / num_qm_components * f_scale
            else:
                e_offset = jnp.mean(e_diff)
                loss_value_e = jnp.mean((e_diff - e_offset) ** 2) * e_scale
                loss_value_f = jnp.mean(fqm_diff ** 2) * f_scale
                num_qm_components = fqm_diff.size

            if "forces_mm" in batch and batch["forces_mm"] is not None:
                fmm_diff = fmm - batch["forces_mm"]
                if "mm_node_mask" in batch:
                    mm_mask = batch["mm_node_mask"].astype(fmm_diff.dtype)[..., None]
                    loss_value_f += jnp.sum((fmm_diff ** 2) * mm_mask) / num_qm_components * f_scale
                else:
                    loss_value_f += jnp.sum(fmm_diff ** 2) / num_qm_components * f_scale

            total_loss = loss_value_e + loss_value_f
            return total_loss, (loss_value_e, loss_value_f)

        (loss, (e_loss, f_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        state = nnx.state((model, optimizer))
        return loss, e_loss, f_loss, state

    def batch_generator():
        while True:
            for b in loader:
                yield jax.device_put(b)

    batch_iter = batch_generator()

    print("Starting training...")
    for step in range(max(args.warmup, 0) + max(args.repeat, 0)):
        step_batch = next(batch_iter)
        t0 = time.time()
        loss, e_loss, f_loss, state = train_step(state, step_batch, e_scale, f_scale)
        jax.block_until_ready(loss)
        t1 = time.time()

        step_type = "Warmup" if step < args.warmup else "Train "
        print(f"[{step_type}] Step {step} | Loss (%):"
              f" {loss*100:.4f} (E: {e_loss*100:.4f}, F: {f_loss*100:.4f}) | Time: {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
