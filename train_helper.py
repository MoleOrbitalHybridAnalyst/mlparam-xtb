from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from etils import epath
from jax import random
from pyscfad.ml.gto import MolePad
from pyscfad.ml.xtb.param import GFN1ParamArray, GFN1MoleParam
from e3nn_jax import Irreps
from pyscfad.ml.xtb.xtb_pad import GFN1XTB
from pyscfad.xtb.data.radii import COV_D3
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges
from pyscfad.xtb.util import atom_to_bas_indices

from mace_jax.modules.models import ScaleShiftMACE

from mace_jax.tools import bundle as bundle_tools
from mace_jax.cli.mace_jax_from_torch import convert_model
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.data.utils import AtomicNumberTable

from constants import A, Bohr, hartree, eV  # pylint: disable=import-error

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


class XTBModel(nnx.Module):
    """MACE-JAX front-end that predicts per-atom XTB parameter scalings."""

    def __init__(
        self,
        mace_model: ScaleShiftMACE,
        xtb_param: GFN1ParamArray,
        basis,
        preserve_sign: bool = True,
        rngs: nnx.Rngs | None = None,
        node_feat_dim: int | None = None,
        max_qm: int | None = None,
        max_mm: int | None = None,
        ew_precision: float = 1e-6,
        scf_conv_tol: float = 1e-6,
        scf_verbose: int = 0,
        max_mm_nbr: int = 512,
        mm_ew_rcut: float = 18.0,
        mm_ew_mesh: Sequence[int] = (80, 80, 80),
        qm_ew_mesh: Sequence[int] = (40, 40, 40),
        n_decoder_layer: int = 1,
    ) -> None:
        self.mace = mace_model
        self.xtb_param = nnx.data(xtb_param)
        self.basis = nnx.data(basis)
        self.preserve_sign = preserve_sign
        self.ew_precision = ew_precision
        self.scf_conv_tol = scf_conv_tol
        self.scf_verbose = scf_verbose
        self.max_mm_nbr = max_mm_nbr
        self.mm_ew_rcut = mm_ew_rcut
        self.mm_ew_mesh = mm_ew_mesh
        self.qm_ew_mesh = qm_ew_mesh


        if max_qm is None or max_mm is None:
            raise ValueError("max_qm and max_mm must be provided for static padding.")
        self.max_qm = int(max_qm)
        self.max_mm = int(max_mm)

        self.atom_fields = ("EN", "arep", "zeff", "gam3", "dipgam", "quadgam")
        self.shell_fields = ("kcn", "selfenergy", "shpoly", "lgam")
        self.max_shells = int(self.basis.nbas)
        self.head_dim = len(self.atom_fields) + len(self.shell_fields) * self.max_shells

        if rngs is None:
            rngs = nnx.Rngs(params=random.key(0))
        self._rngs = rngs

        scalar_indices = scalar_node_feature_indices(mace_model)
        if scalar_indices is None:
            self._scalar_feature_indices = None
            if node_feat_dim is None:
                raise ValueError("node_feat_dim must be provided when the MACE model lacks product irreps.")
        else:
            self._scalar_feature_indices = scalar_indices
            scalar_dim = int(self._scalar_feature_indices.shape[0])
            if scalar_dim == 0:
                raise ValueError("MACE model does not expose any scalar node features.")
            if node_feat_dim is None:
                node_feat_dim = scalar_dim
            elif node_feat_dim != scalar_dim:
                raise ValueError(
                    "Provided node_feat_dim must match the scalar channels exposed by the MACE model."
                )
        
        layers = []
        for _ in range(n_decoder_layer - 1):
            layers.append(nnx.silu)
            layers.append(nnx.Linear(node_feat_dim, node_feat_dim, rngs=self._rngs))
        layers.append(nnx.silu)
        layers.append(nnx.Linear(node_feat_dim, self.head_dim, rngs=self._rngs))
        self.decoder = nnx.Sequential(*layers)
        decoder_linear = self.decoder.layers[-1]
        decoder_linear.kernel.value = decoder_linear.kernel.value * 0.

        self.global_names = ("kf", "kEN")
        self.global_factors = nnx.Param(
            jnp.ones((len(self.global_names),), dtype=jnp.float64)
        )
        self.offset = nnx.Param(jnp.zeros((), dtype=jnp.float64))

        self.k_shlpr_diag_factors = nnx.Param(jnp.ones(xtb_param.k_shlpr.shape[0], dtype=jnp.float64))

        if self.preserve_sign:
            self.global_factors.value = jnp.full_like(self.global_factors.value, 0.5413248)
            self.k_shlpr_diag_factors.value = jnp.full_like(self.k_shlpr_diag_factors.value, 0.5413248)
            decoder_linear.bias.value = jnp.ones_like(decoder_linear.bias.value) * 0.5413248
        else:
            decoder_linear.bias.value = jnp.ones_like(decoder_linear.bias.value)

        self.mm_radii_table = nnx.data(jnp.asarray(COV_D3, dtype=jnp.float64))

    def __call__(self, batchdict: dict[str, jnp.ndarray]):
        mace_out = self.mace(batchdict, compute_node_feats=True)
        node_feats = mace_out["node_feats"]
        if node_feats is None:
            raise RuntimeError("MACE model must return node_feats for parameter head.")

        if self._scalar_feature_indices is not None:
            node_feats = node_feats[:, self._scalar_feature_indices]

        atomwise_raw = self.decoder(node_feats)
        if self.preserve_sign:
            atomwise = jax.nn.softplus(atomwise_raw)
            gfactors = jax.nn.softplus(self.global_factors.value)
            k_shlpr_factors = jax.nn.softplus(self.k_shlpr_diag_factors.value)
        else:
            atomwise = atomwise_raw
            gfactors = self.global_factors.value
            k_shlpr_factors = self.k_shlpr_diag_factors.value
        base_param = self.xtb_param
        # NOTE that ksp != (ks + kp) / 2 but close
        # so not exact GFN1xTB energy may be recovered
        k_shlpr_diag = jnp.diagonal(base_param.k_shlpr) * k_shlpr_factors
        k_shlpr = 0.5 * (k_shlpr_diag[:, None] + k_shlpr_diag[None, :])
        xtb_param = replace(base_param, k_shlpr=k_shlpr)


        # pack per-graph tensors to fixed shapes and run vmapped XTB
        packed = self._pack_batch(batchdict, atomwise)
        e_xtb = jax.vmap(
            self._xtb_energy_single,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
        )(
            packed["zqm"],
            packed["Rqm"],
            packed["zmm"],
            packed["Rmm"],
            packed["qqm"],
            packed["qmm"],
            packed["cell"],
            packed["atomwise"],
            packed["mask_qm"],
            packed["mask_mm"],
            gfactors,
            xtb_param,
        )
        e_mace = mace_out["interaction_energy"]
        return e_xtb + e_mace + self.offset.value

    def _pack_batch(self, batchdict, atomwise):
        """Pad variable-size molecules to fixed shapes for vmapping."""
        ptr = jnp.asarray(batchdict["ptr"])
        ptr_mm = jnp.asarray(batchdict["ptr_mm"])
        n_graph = ptr.shape[0] - 1
        max_qm = self.max_qm
        max_mm = self.max_mm

        idx_q = jnp.arange(max_qm)
        idx_m = jnp.arange(max_mm)

        def gather_q(i):
            p1 = ptr[i]
            p2 = ptr[i + 1]
            nq = p2 - p1
            sel = jnp.clip(p1 + idx_q, 0, batchdict["z"].shape[0] - 1)
            mask = idx_q < nq
            z = jnp.where(mask, batchdict["z"][sel], 0)
            R = jnp.where(mask[:, None], batchdict["positions"][sel], 0.)
            q = jnp.where(mask, batchdict["q"][sel], 0.0)
            aw = jnp.where(mask[:, None], atomwise[sel], 0.0)
            maskf = mask.astype(jnp.float32)
            return z, R, q, aw, maskf

        def gather_m(i):
            q1 = ptr_mm[i]
            q2 = ptr_mm[i + 1]
            nm = q2 - q1
            sel = jnp.clip(q1 + idx_m, 0, batchdict["z_mm"].shape[0] - 1)
            mask = idx_m < nm
            z = jnp.where(mask, batchdict["z_mm"][sel], 0)
            R = jnp.where(mask[:, None], batchdict["positions_mm"][sel], 0.)
            q = jnp.where(mask, batchdict["q_mm"][sel], 0.0)
            maskf = mask.astype(jnp.float32)
            return z, R, q, maskf

        zqm, Rqm, qqm, atomwise_p, mask_qm = jax.vmap(gather_q)(jnp.arange(n_graph))
        zmm, Rmm, qmm, mask_mm = jax.vmap(gather_m)(jnp.arange(n_graph))
        cell = batchdict["cell"][:n_graph]

        return {
            "zqm": zqm,
            "Rqm": Rqm,
            "zmm": zmm,
            "Rmm": Rmm,
            "qqm": qqm,
            "qmm": qmm,
            "cell": cell,
            "atomwise": atomwise_p,
            "mask_qm": mask_qm,
            "mask_mm": mask_mm,
        }

    def _split_atomwise(self, atomwise: jnp.ndarray):
        atom_part = atomwise[:, : len(self.atom_fields)]
        shell_part = atomwise[:, len(self.atom_fields) :]
        shell_part = shell_part.reshape(
            atomwise.shape[0], self.max_shells, len(self.shell_fields)
        )
        return atom_part, shell_part

    def _apply_global(self, param: GFN1ParamArray, gfactors: jnp.ndarray) -> GFN1ParamArray:
        kf = param.kf * gfactors[0]
        kEN = param.kEN * gfactors[1]
        return replace(param, kf=kf, kEN=kEN)

    def _apply_atomwise(
        self,
        mol: MolePad,
        param: GFN1MoleParam,
        numbers: jnp.ndarray,
        atomwise: jnp.ndarray,
        mask_qm: jnp.ndarray,
    ) -> GFN1MoleParam:
        atom_part, shell_part = self._split_atomwise(atomwise)
        atom_mask = mask_qm
        shell_mask = jnp.asarray(self.basis.mask_shl[jnp.asarray(numbers, dtype=int)])
        shell_mask = shell_mask & (atom_mask[..., None] > 0)
        shell_part = jnp.where(shell_mask[..., None], shell_part, 1.0)

        flat = lambda idx: shell_part[..., idx].reshape(-1)

        atom_part = jnp.where(atom_mask[..., None] > 0, atom_part, 1.0)
        dipgam = None if param.dipgam is None else param.dipgam * atom_part[:, 4]
        quadgam = None if param.quadgam is None else param.quadgam * atom_part[:, 5]

        return replace(
            param,
            EN=param.EN * atom_part[:,0],
            arep=param.arep * atom_part[:, 1],
            zeff=param.zeff * atom_part[:, 2],
            gam3=param.gam3 * atom_part[:, 3],
            dipgam=dipgam,
            quadgam=quadgam,
            kcn=param.kcn * flat(0),
            selfenergy=param.selfenergy * flat(1),
            shpoly=param.shpoly * flat(2),
            lgam=param.lgam * flat(3),
        )

    def _xtb_energy_single(
        self,
        zqm: jnp.ndarray,
        Rqm: jnp.ndarray,
        zmm: jnp.ndarray,
        Rmm: jnp.ndarray,
        qqm: jnp.ndarray,
        qmm: jnp.ndarray,
        cell: jnp.ndarray,
        atomwise: jnp.ndarray,
        mask_qm: jnp.ndarray,
        mask_mm: jnp.ndarray,
        gfactors: jnp.ndarray,
        xtb_param: GFN1ParamArray,
    ) -> jnp.ndarray:
        coords_bohr = jnp.asarray(Rqm * A / Bohr)
        mm_coords_bohr = jnp.asarray(Rmm * A / Bohr)
        cell_bohr = jnp.asarray(cell * A / Bohr)

        charge = jnp.round(jnp.sum(qqm * mask_qm)).astype(int)
        mol = MolePad(
            jnp.asarray(zqm, dtype=jnp.int32),
            coords_bohr,
            basis=self.basis,
            verbose=self.scf_verbose,
            trace_coords=True,
            charge=charge,
        )

        param_arr = self._apply_global(xtb_param, gfactors)
        param_mol = param_arr.to_mol_param(mol)
        param_mol = self._apply_atomwise(mol, param_mol, zqm, atomwise, mask_qm)

        mf = GFN1XTB(mol, param_mol)
        mm_radii = self.mm_radii_table[jnp.asarray(zmm, dtype=jnp.int32)]
        mm_radii = mm_radii * mask_mm
        mf = add_mm_charges(
            mf,
            mm_coords_bohr,
            cell_bohr,
            jnp.asarray(qmm * mask_mm),
            jnp.asarray(mm_radii),
            max_mm_nbr=self.max_mm_nbr,
            mm_ew_rcut=self.mm_ew_rcut,
            mm_ew_mesh=self.mm_ew_mesh,
            qm_ew_mesh=self.qm_ew_mesh,
            ew_precision=self.ew_precision,
            unit="Bohr",
            pbcqm=True,
        )
        mf.diis = "qbroyden"
        mf.conv_tol = self.scf_conv_tol
        mf.diis_damp = 0.6

        atm_to_bas = jnp.asarray(atom_to_bas_indices(mol))
        nbas_per_atom = jnp.bincount(atm_to_bas, length=mol.natm)
        safe_nbas = jnp.where(nbas_per_atom > 0, nbas_per_atom, 1)
        q0 = (qqm * mask_qm)[atm_to_bas] / safe_nbas[atm_to_bas]

        q0_parts = [q0]
        if param_mol.dipgam is not None:
            q0_parts.append(jnp.zeros(mol.natm * 3, dtype=q0.dtype))
        if param_mol.quadgam is not None:
            q0_parts.append(jnp.zeros(mol.natm * 9, dtype=q0.dtype))
        q0 = jnp.concatenate(q0_parts)

        energy = mf.kernel(q0=q0)

        return jnp.asarray(energy) * hartree / eV


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
    def __init__(self, ckpt_dir, keep_latest=1, keep_best=1, save_stride=200):
        self.ckpt_dir = epath.Path(ckpt_dir).resolve()

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

__all__ = [
    "XTBModel",
    "scalar_node_feature_indices",
    "energy_loss",
    "force_loss",
    "Checkpointer",
    "load_mace_module",
]
