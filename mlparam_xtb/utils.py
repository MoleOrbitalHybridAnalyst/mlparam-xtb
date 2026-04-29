from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from etils import epath
from e3nn_jax import Irreps

if TYPE_CHECKING:
    from .models import XTBModel
    from .data import QMMMDataset

from mace_jax.tools import bundle as bundle_tools
from mace_jax.cli.mace_jax_from_torch import convert_model
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.data.utils import AtomicNumberTable

def load_mace_module(model_path: str):
    # Try native JAX bundle first
    try:
        bundle = bundle_tools.load_model_bundle(model_path, dtype="float64")
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

def scalar_node_feature_indices(mace_model: Any) -> jnp.ndarray | None:
    """Return the scalar-node indices visible in the MACE products.

    Returns ``None`` if ``mace_model`` does not expose ``products`` or if the
    products do not specify ``target_irreps`` (e.g. dummy modules)."""

    products = getattr(mace_model, "products", None)
    if products is None:
        return None

    indices: list[int] = []
    offset = 0
    for product in products:
        target_irreps = getattr(product, "target_irreps", None)
        if target_irreps is None:
            return None
        for mul_ir in Irreps(target_irreps):
            chunk = mul_ir.dim
            if mul_ir.ir.is_scalar():
                indices.extend(range(offset, offset + chunk))
            offset += chunk
    if not indices:
        raise ValueError("MACE model does not expose any scalar node features.")
    return jnp.asarray(indices, dtype=jnp.int32)

def energy_loss(epred, eref, zqm, bqm, scale: str | float = "var", eps: float = 1e-8):
    """Composition-aware energy loss using JAX arrays."""
    def composition_signature(z):
        return tuple([len(z)] + jnp.sort(z).tolist())

    nbatch = epred.shape[0]
    comp_dict: dict[tuple, list[int]] = {}
    for mol_idx in range(nbatch):
        sig = composition_signature(zqm[bqm == mol_idx])
        comp_dict.setdefault(sig, []).append(mol_idx)

    total = 0.0
    count = 0
    for mol_indices in comp_dict.values():
        idx = jnp.asarray(mol_indices, dtype=jnp.int32)
        ep = epred[idx]
        er = eref[idx]
        if isinstance(scale, str):
            var_c = jnp.var(er) if er.size > 1 else eps
        else:
            var_c = 1.0 / scale
        total += jnp.mean((ep - er) ** 2) / var_c
        count += er.size
    return total / count

def force_loss(f_pred, f_ref, scale: str | float = "var"):
    if isinstance(scale, str):
        if scale == "var":
            sc = 1.0 / jnp.var(f_ref)
        else:
            raise NotImplementedError
    else:
        sc = scale
    return jnp.mean((f_pred - f_ref) ** 2) * sc

class Checkpointer:
    """
    Manages model checkpoints using orbax.checkpoint.
    Maintains 'latest', 'best', and 'periodic' histories across an entire run.
    """
    def __init__(self, ckpt_dir, keep_latest=1, keep_best=1, save_stride=200, erase=False):
        self.ckpt_dir = epath.Path(ckpt_dir).resolve()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if erase:
            ocp.test_utils.erase_and_create_empty(self.ckpt_dir)

        self.latest_mngr = ocp.CheckpointManager(
            self.ckpt_dir / "latest",
            options=ocp.CheckpointManagerOptions(max_to_keep=keep_latest, save_interval_steps=1)
        )
        
        self.best_mngr = ocp.CheckpointManager(
            self.ckpt_dir / "best",
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep_best,
                best_fn=lambda m: m['val_loss'],
                best_mode='min',
                keep_checkpoints_without_metrics=False,
                save_interval_steps=1,
            )
        )
        
        if save_stride is not None and save_stride > 0:
            self.periodic_mngr = ocp.CheckpointManager(
                self.ckpt_dir / "periodic",
                options=ocp.CheckpointManagerOptions(max_to_keep=None, save_interval_steps=save_stride)
            )
        else:
            self.periodic_mngr = None

    def save(self, step, state, val_loss=None):
        save_args = ocp.args.StandardSave(state)
        
        self.latest_mngr.save(step, args=save_args)
        self.latest_mngr.wait_until_finished()
        
        if val_loss is not None:
            self.best_mngr.save(step, args=save_args, metrics={'val_loss': float(val_loss)})
            self.best_mngr.wait_until_finished()
            
        if self.periodic_mngr is not None:
            self.periodic_mngr.save(step, args=save_args)
            self.periodic_mngr.wait_until_finished()

    def load(self, state, load="best"):
        if load == "best":
            mngr = self.best_mngr
            step = mngr.best_step()
        elif load == "latest":
            mngr = self.latest_mngr
            step = mngr.latest_step()
        else:
            mngr = self.periodic_mngr
            step = int(load)
        return mngr.restore(step, args=ocp.args.StandardRestore(state)), step

def precompute_init_guess(
    model: "XTBModel",
    dataset: "QMMMDataset",
    batch_size: int = 1,
) -> None:
    """Populate `shell_charges`/`atom_dips`/`atom_quads` on every sample in
    `dataset` by running SCF once with the model's current parameters.

    Run this at the start of training, then set `model.scf_init_guess = "from_data"`
    so subsequent SCFs warm-start from the converged charges. Re-run periodically
    if model parameters drift far from the precomputed regime.
    """
    from .data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, pad=True)
    sample_idx = 0
    n_samples = len(dataset.samples)
    for batchdict in loader:
        sc, ad, aq = model.precompute_charges(batchdict)
        sc = np.asarray(sc)
        ad = np.asarray(ad)
        aq = np.asarray(aq)
        for g in range(sc.shape[0]):
            if sample_idx >= n_samples:
                return
            sample = dataset.samples[sample_idx]
            n_q = sample.num_nodes
            sample.shell_charges = sc[g, :n_q].copy()
            sample.atom_dips = ad[g, :n_q].copy()
            sample.atom_quads = aq[g, :n_q].copy()
            sample_idx += 1


def precompute_cuint_plan(model: "XTBModel", dataset: "QMMMDataset") -> None:
    """Build a cuint plan for `dataset` and cache it on the dataset, so the
    `DataLoader` will inject it into every batch and the model dispatches xTB
    integrals to the GPU backend.

    Assumes a single chemical composition across the dataset (true for a
    QMMMDataset built from one npz file).
    """
    from pyscfad.experimental.moleintor_cuint import _cuint, cuint_create_plan
    from pyscfad.ml.gto import MolePad

    if not _cuint:
        raise RuntimeError(
            "pyscfad cuint backend not available; rebuild pyscfad with CUDA support."
        )
    if not dataset.samples:
        raise ValueError("Cannot precompute cuint plan for empty dataset.")

    z0 = np.asarray(dataset.samples[0].z, dtype=np.int32)
    for s in dataset.samples[1:]:
        z = np.asarray(s.z, dtype=np.int32)
        if z.shape != z0.shape or not np.array_equal(z, z0):
            raise NotImplementedError(
                "precompute_cuint_plan requires all samples to share the same "
                "atomic-number sequence; mixed-composition datasets need per-batch "
                "plan merging which is not yet implemented."
            )

    numbers = np.zeros(model.max_qm, dtype=np.int32)
    numbers[: z0.shape[0]] = z0
    coords = np.zeros((model.max_qm, 3), dtype=np.float64)
    mol = MolePad(numbers, coords, basis=model.basis, verbose=0)
    dataset.cuint_plan = cuint_create_plan(mol)


__all__ = [
    "scalar_node_feature_indices",
    "energy_loss",
    "force_loss",
    "Checkpointer",
    "load_mace_module",
    "precompute_init_guess",
    "precompute_cuint_plan",
]
