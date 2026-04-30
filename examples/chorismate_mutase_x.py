''' accelerated chorismate_mutase.py via 
    1. pre-computed SCF initial guess
    2. GPU-accelerated integrals
'''
from __future__ import annotations

import argparse
import time
import optax
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.gto import make_basis_array
from pyscfad.ml.xtb.param import make_param_array

from mlparam_xtb.data import QMMMDataset, DataLoader
from mlparam_xtb.models import XTBModel
from mlparam_xtb.utils import (scalar_node_feature_indices, 
                               load_mace_module,
                               precompute_init_guess, 
                               precompute_cuint_plan )



def parse_args():
    p = argparse.ArgumentParser(description="Smoke test XTBModel with real data/model")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(
            "./chorismate_mutase_deps/neb_data.npz"
        ),
        help="NPZ file with QM/MM data (units in a.u.)",
    )
    p.add_argument(
        "--train_list",
        type=Path,
        default=Path("./chorismate_mutase_deps/neb.train.list.0"),
        help="Txt file of frame indices to load for training",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "./chorismate_mutase_deps/maceoff24_pretrained.torch.model"
        ),
        help="MACE-JAX bundle dir or ckpt (config.json+params.msgpack). ",
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
        default=50,
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

    try:
        mace_module, z_table, cutoff = load_mace_module(args.model)
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

    # Scan dataset for energy and force ranges
    energies = []
    forces = []
    for s in dataset.samples:
        if s.energy is not None:
            energies.append(np.max(s.energy))
            energies.append(np.min(s.energy))
        if s.forces is not None:
            forces.append(np.max(s.forces))
            forces.append(np.min(s.forces))
        if s.forces_mm is not None:
            forces.append(np.max(s.forces_mm))
            forces.append(np.min(s.forces_mm))

    e_scale = float(1.0 / max(np.ptp(energies) ** 2, 1e-8))
    f_scale = float(1.0 / max(np.ptp(forces) ** 2, 1e-8))
    print(f"e_scale: {e_scale:.4e}, f_scale: {f_scale:.4e}")


    basis = make_basis_array(xtb_basis.get_basis_filename(), max_number=max_z)
    param = make_param_array(basis, max_number=max_z)
    param.dipgam = jnp.array(param.gam)
    param.quadgam = jnp.array(param.gam)


    rngs = nnx.Rngs(params=jax.random.key(0))
    # determine node scalar feature dim
    mace_out = mace_module(batch, compute_node_feats=True)
    scalar_indices = scalar_node_feature_indices(mace_module)
    node_feat_dim = int(scalar_indices.shape[0])

    model = XTBModel(
        mace_model=mace_module,
        xtb_param=param,
        basis=basis,
        rngs=rngs,
        node_feat_dim=node_feat_dim,
        max_qm=max_qm,
        max_mm=max_mm,
        # MM mesh should be roughly 1 grid per 1 angstrom of total box
        mm_ew_mesh=(80, 80, 80),
        # QM mesh is system independent
        qm_ew_mesh=(40, 40, 40),
        n_decoder_layer=2,
        scf_init_guess="atomic_charges",
        scf_diis_damp=0.9,
    )
    precompute_cuint_plan(model, dataset) # write cuint plan to use GPU integrals
    precompute_init_guess(model, dataset, batch_size=3)  # write converged SCF into data
    model.scf_init_guess = "from_data"  # subsequent SCFs warm-start from data

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
