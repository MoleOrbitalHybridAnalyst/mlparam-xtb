from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from pyscfad.ml.gto import make_basis_array
from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.xtb.param import make_param_array

from mace_jax.data.utils import AtomicNumberTable
from mace_jax.modules import ScaleShiftMACE, RealAgnosticInteractionBlock
from mace_jax.nnx_config import ConfigVar
from mace_jax.nnx_utils import state_to_pure_dict
from mlparam_xtb.data import QMMMData, _collate
from mlparam_xtb.models import XTBModel
from mlparam_xtb.utils import scalar_node_feature_indices

from e3nn_jax import Irreps

from random import seed, shuffle
seed(123123)


def _merge_state_dicts(base: dict | None, updates: dict | None) -> dict | None:
    if base is None:
        return updates
    if updates is None:
        return base
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_state_dicts(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _iter_leaves(tree, prefix=()):
    if isinstance(tree, dict):
        for key in sorted(tree.keys()):
            yield from _iter_leaves(tree[key], prefix + (key,))
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            yield from _iter_leaves(value, prefix + (idx,))
    else:
        yield prefix, tree


def _tree_get(tree, path):
    out = tree
    for key in path:
        out = out[key]
    return out


def _tree_set(tree, path, value):
    if not path:
        return value
    key = path[0]
    if isinstance(tree, dict):
        updated = dict(tree)
        updated[key] = _tree_set(tree[key], path[1:], value)
        return updated
    if isinstance(tree, list):
        updated = list(tree)
        updated[key] = _tree_set(tree[key], path[1:], value)
        return updated
    if isinstance(tree, tuple):
        updated = list(tree)
        updated[key] = _tree_set(tree[key], path[1:], value)
        return type(tree)(updated)
    raise TypeError(f"Unsupported tree node for set: {type(tree)}")


def _sample_flat_indices(size: int, max_items: int) -> list[int]:
    if size <= 0:
        return []
    if size <= max_items:
        return list(range(size))
    step = size / max_items
    return [int(i * step) for i in range(max_items)]


def _path_to_str(path) -> str:
    return "/".join(str(p) for p in path)


def make_sample(z_table):
    """Build a tiny water-like QM with two MM atoms."""
    zqm = jnp.array([8, 1, 1], dtype=jnp.int32)
    Rqm = jnp.array(
        [
            [0.0000, 0.0000, 0.1000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9266, 0.0000],
        ]
    )
    zmm = jnp.array([1, 6], dtype=jnp.int32)
    Rmm = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
        ]
    )
    a = jnp.diag(jnp.array([10, 11, 12])) # Angstrom
    qqm = jnp.zeros((3,))
    qmm = jnp.array([2., -2.])
    return QMMMData.from_raw(
        zqm=zqm,
        Rqm=Rqm,
        zmm=zmm,
        Rmm=Rmm,
        a=a,
        E=None,
        Fqm=None,
        Fmm=None,
        qqm=qqm,
        qmm=qmm,
        z_table=z_table,
        cutoff=5.0,
    )

def make_sample_two(z_table):
    """Build two-water-like QM with three MM atoms."""
    zqm = jnp.array([8, 1, 1, 8, 1, 1], dtype=jnp.int32)
    Rqm = jnp.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9266, 0.0000],
            [0.0000, 0.0000, 3.0000],
            [0.9572, 0.0000, 3.0000],
            [-0.2390, 0.9266, 3.0000],
        ]
    )
    zmm = jnp.array([1, 6, 8], dtype=jnp.int32)
    Rmm = jnp.array(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    a = jnp.diag(jnp.array([13, 12, 14]))
    qqm = jnp.zeros((6,))
    qmm = jnp.array([-2.0, 4.0, -2.0])
    return QMMMData.from_raw(
        zqm=zqm,
        Rqm=Rqm,
        zmm=zmm,
        Rmm=Rmm,
        a=a,
        E=None,
        Fqm=None,
        Fmm=None,
        qqm=qqm,
        qmm=qmm,
        z_table=z_table,
        cutoff=5.0,
    )

def make_dummy_sample(z_table):
    """Build a dummy sample"""
    zqm = jnp.array([1], dtype=jnp.int32)
    Rqm = jnp.zeros((1,3), dtype=jnp.float64)
    zmm = jnp.array([1], dtype=jnp.int32)
    Rmm = jnp.zeros((1,3), dtype=jnp.float64)
    a = jnp.eye(3)
    qqm = jnp.array([0.])
    qmm = jnp.array([0.])
    return QMMMData.from_raw(
        zqm=zqm,
        Rqm=Rqm,
        zmm=zmm,
        Rmm=Rmm,
        a=a,
        E=None,
        Fqm=None,
        Fmm=None,
        qqm=qqm,
        qmm=qmm,
        z_table=z_table,
        cutoff=5.0,
    )

def main():
    # Basis / parameters
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)

    # Atomic number table from QM atoms
    z_table = AtomicNumberTable([1, 8])

    # Build two samples and collate (note the last sample will always be ignored)
    samples = [make_sample(z_table), make_sample_two(z_table), make_dummy_sample(z_table)]
    batch = _collate(samples)
    batch = jax.device_put(batch)

    max_qm = max(s.num_nodes for s in samples)
    max_mm = max(s.n_mm for s in samples)

    param.dipgam = jnp.array(param.gam)
    param.quadgam = jnp.array(param.gam)

    rngs = nnx.Rngs(0)
    # Build a tiny MACE model (no pretrained weights)
    mace = ScaleShiftMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=RealAgnosticInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        atomic_energies=jnp.zeros((len(z_table),), dtype=jnp.float64),
        atomic_numbers=tuple(z_table.zs),
        num_interactions=2,
        num_elements=len(z_table),
        hidden_irreps=Irreps("16x0e+16x1o"),
        MLP_irreps=Irreps("16x0e"),
        avg_num_neighbors=4.0,
        rngs=rngs,
    )

    mace_out = mace(batch, compute_node_feats=True)
    scalar_indices = scalar_node_feature_indices(mace)
    node_feat_dim = (
        int(scalar_indices.shape[0])
        if scalar_indices is not None
        else int(mace_out["node_feats"].shape[-1])
    )

    model = XTBModel(
        mace_model=mace,
        xtb_param=param,
        basis=basis,
        rngs=rngs,
        node_feat_dim=node_feat_dim,
        max_qm=max_qm,
        max_mm=max_mm,
        ew_precision=1e-7,
        scf_conv_tol=1e-8,
        mm_ew_mesh=(80, 80, 80),
        qm_ew_mesh=(40, 40, 40),
        n_decoder_layer=2,
    )

    # add some noise to perturb xtb parameters
    decoder_linear = model.decoder.layers[-1]
    key = jax.random.key(1)
    decoder_linear.kernel.value = (
        jax.random.normal(
            key,
            decoder_linear.kernel.shape,
            dtype=decoder_linear.kernel.dtype,
        )
        * 1e-1
    )

    # === Gradient checks ===
    def energy_from_positions(pos_qm, pos_mm):
        batch_local = dict(batch)
        batch_local["positions"] = pos_qm
        batch_local["positions_mm"] = pos_mm
        return jnp.sum(model(batch_local))

    g_qm, g_mm = nnx.grad(energy_from_positions, argnums=(0, 1))(
        batch["positions"], batch["positions_mm"]
    )

    energy_from_positions_jit = nnx.jit(energy_from_positions)

    eps_coord = 1e-4
    qm_flat = batch["positions"].reshape(-1)
    mm_flat = batch["positions_mm"].reshape(-1)
    qm_idx = _sample_flat_indices(qm_flat.size, 10)
    mm_idx = _sample_flat_indices(mm_flat.size, 10)

    print("\nQM coordinate gradient check (finite diff, sampled):")
    for idx in qm_idx:
        pos_plus = qm_flat.at[idx].add(eps_coord).reshape(batch["positions"].shape)
        pos_minus = qm_flat.at[idx].add(-eps_coord).reshape(batch["positions"].shape)
        e_plus = energy_from_positions_jit(pos_plus, batch["positions_mm"])
        e_minus = energy_from_positions_jit(pos_minus, batch["positions_mm"])
        fd = (e_plus - e_minus) / (2 * eps_coord)
        grad = g_qm.reshape(-1)[idx]
        atom = idx // 3
        comp = idx % 3
        err = jnp.abs(fd - grad)
        print(
            f"  atom {atom:2d} comp {comp}: grad={float(grad): .6e} "
            f"fd={float(fd): .6e} |err|={float(err): .2e}"
        )

    print("\nMM coordinate gradient check (finite diff, sampled):")
    for idx in mm_idx:
        pos_plus = mm_flat.at[idx].add(eps_coord).reshape(batch["positions_mm"].shape)
        pos_minus = mm_flat.at[idx].add(-eps_coord).reshape(batch["positions_mm"].shape)
        e_plus = energy_from_positions_jit(batch["positions"], pos_plus)
        e_minus = energy_from_positions_jit(batch["positions"], pos_minus)
        fd = (e_plus - e_minus) / (2 * eps_coord)
        grad = g_mm.reshape(-1)[idx]
        atom = idx // 3
        comp = idx % 3
        err = jnp.abs(fd - grad)
        print(
            f"  atom {atom:2d} comp {comp}: grad={float(grad): .6e} "
            f"fd={float(fd): .6e} |err|={float(err): .2e}"
        )

    # Parameter gradient check (sampled entries)
    def energy_from_params(params_pure, graphdef, config_pure):
        state_local = _merge_state_dicts(config_pure, params_pure)
        energy_out, _ = graphdef.apply(state_local)(batch)
        return jnp.sum(energy_out)

    graphdef, state = nnx.split(model)
    params_state, config_state, rest_state = nnx.split_state(
        state, nnx.Param, ConfigVar, ...
    )
    if rest_state:
        config_state = nnx.merge_state(config_state, rest_state)
    params_pure = state_to_pure_dict(params_state)
    config_pure = state_to_pure_dict(config_state) if config_state else None

    def loss_fn(m):
        return jnp.sum(m(batch))

    param_grads_state = nnx.grad(loss_fn)(model)
    param_grads = state_to_pure_dict(param_grads_state)

    def sum_forces_fn(m):
        def e_fn(pos, pos_mm):
            batch_local = dict(batch)
            batch_local["positions"] = pos
            batch_local["positions_mm"] = pos_mm
            return jnp.sum(m(batch_local))
        g_qm, g_mm = nnx.grad(e_fn, argnums=(0,1))(batch["positions"], batch["positions_mm"])
        return jnp.sum(g_qm**2) + jnp.sum(g_mm**2)

    param_force_grads_state = nnx.grad(sum_forces_fn)(model)
    param_force_grads = state_to_pure_dict(param_force_grads_state)

    energy_from_params_jit = nnx.jit(energy_from_params)

    def forces_from_params(params_pure, graphdef, config_pure):
        def e_fn(pos, pos_mm):
            batch_local = dict(batch)
            batch_local["positions"] = pos
            batch_local["positions_mm"] = pos_mm
            state_local = _merge_state_dicts(config_pure, params_pure)
            energy_out, _ = graphdef.apply(state_local)(batch_local)
            return jnp.sum(energy_out)
        g_qm, g_mm = nnx.grad(e_fn, argnums=(0,1))(batch["positions"], batch["positions_mm"])
        return jnp.sum(g_qm**2) + jnp.sum(g_mm**2)

    forces_from_params_jit = nnx.jit(forces_from_params)

    eps_param = 1e-4
    print("\nParameter gradient check (finite diff, sampled):")
    checked = 0
    all_leaves = list(_iter_leaves(params_pure))
    shuffle(all_leaves)
    all_leaves.sort(key=lambda x: 0 if "decoder" in x[0] else 1)
    for path, arr in all_leaves:
        if checked >= 20:
            break
        if not isinstance(arr, jnp.ndarray) or not jnp.issubdtype(arr.dtype, jnp.inexact):
            continue
        if arr.size == 0:
            continue

        param_var = _tree_get(params_state, path)
        if getattr(param_var, 'is_mutable', True) is False:
            continue

        idx = 0
        arr_flat = arr.reshape(-1)
        grad_arr = _tree_get(param_grads, path)
        grad_val = grad_arr.reshape(-1)[idx]

        grad_force_arr = _tree_get(param_force_grads, path)
        grad_force_val = grad_force_arr.reshape(-1)[idx]

        arr_plus = arr_flat.at[idx].add(eps_param).reshape(arr.shape)
        arr_minus = arr_flat.at[idx].add(-eps_param).reshape(arr.shape)
        params_plus = _tree_set(params_pure, path, arr_plus)
        params_minus = _tree_set(params_pure, path, arr_minus)

        e_plus = energy_from_params_jit(params_plus, graphdef, config_pure)
        e_minus = energy_from_params_jit(params_minus, graphdef, config_pure)
        fd = (e_plus - e_minus) / (2 * eps_param)
        err = jnp.abs(fd - grad_val)
        print(
            f"  {_path_to_str(path)}[{idx}] (Energy) grad={float(grad_val): .6e} "
            f"fd={float(fd): .6e} |err|={float(err): .2e}"
        )

        f_plus = forces_from_params_jit(params_plus, graphdef, config_pure)
        f_minus = forces_from_params_jit(params_minus, graphdef, config_pure)
        f_fd = (f_plus - f_minus) / (2 * eps_param)
        f_err = jnp.abs(f_fd - grad_force_val)
        print(
            f"  {_path_to_str(path)}[{idx}] (Force ) grad={float(grad_force_val): .6e} "
            f"fd={float(f_fd): .6e} |err|={float(f_err): .2e}"
        )
        checked += 1


if __name__ == "__main__":
    main()
