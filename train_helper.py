from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random

from pyscfad import numpy as anp
from pyscfad.ml.gto import MolePad
from pyscfad.ml.xtb.param import GFN1ParamArray, GFN1MoleParam
from pyscfad.ml.xtb.xtb_pad import GFN1XTB
from pyscfad.xtb.data.radii import COV_D3
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges
from pyscfad.xtb.util import atom_to_bas_indices

from mace_jax.modules.models import ScaleShiftMACE

from constants import A, Bohr, hartree, eV  # pylint: disable=import-error


def loop_molecule_from_batch(batchdict, atomwise):
    """Yield per-molecule slices from a batched dictionary."""
    ptr = jnp.asarray(batchdict["ptr"])
    ptr_mm = jnp.asarray(batchdict["ptr_mm"])
    for i in range(ptr.shape[0] - 1):
        p1, p2 = int(ptr[i]), int(ptr[i + 1])
        q1, q2 = int(ptr_mm[i]), int(ptr_mm[i + 1])
        yield (
            batchdict["z"][p1:p2],
            batchdict["positions"][p1:p2],
            batchdict["z_mm"][q1:q2],
            batchdict["positions_mm"][q1:q2],
            batchdict["q"][p1:p2],
            batchdict["q_mm"][q1:q2],
            batchdict["cell"][i],
            atomwise[p1:p2],
        )


class XTBModel(nnx.Module):
    """MACE-JAX front-end that predicts per-atom XTB parameter scalings."""

    def __init__(
        self,
        mace_model: ScaleShiftMACE,
        xtb_param: GFN1ParamArray,
        basis,
        preserve_sign: bool = True,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.mace = mace_model
        self.xtb_param = xtb_param
        self.basis = basis
        self.preserve_sign = preserve_sign

        self.atom_fields = ("arep", "zeff", "gam", "gam3")
        self.shell_fields = ("kcn", "selfenergy", "shpoly", "lgam")
        self.max_shells = int(self.basis.nbas)
        self.head_dim = len(self.atom_fields) + len(self.shell_fields) * self.max_shells

        if rngs is None:
            rngs = nnx.Rngs(params=random.key(0))

        self.decoder = nnx.Sequential(
            nnx.SiLU(rngs=rngs),
            nnx.Linear(self.head_dim, rngs=rngs),
        )

        self.global_names = ("kf", "kEN", "kcn_d3")
        self.global_factors = nnx.Param(
            jnp.ones((len(self.global_names),), dtype=jnp.float64)
        )
        self.offset = nnx.Param(jnp.zeros((), dtype=jnp.float64))

        self.mm_radii_table = jnp.asarray(COV_D3, dtype=jnp.float64)

    def __call__(self, batchdict: dict[str, jnp.ndarray]):
        mace_out = self.mace(batchdict, compute_node_feats=True)
        node_feats = mace_out["node_feats"]
        if node_feats is None:
            raise RuntimeError("MACE model must return node_feats for parameter head.")

        atomwise_raw = self.decoder(node_feats)
        if self.preserve_sign:
            atomwise = jax.nn.softplus(atomwise_raw)
            gfactors = jax.nn.softplus(self.global_factors.value)
        else:
            atomwise = atomwise_raw
            gfactors = self.global_factors.value

        # pack per-graph tensors to fixed shapes and run vmapped XTB
        packed = self._pack_batch(batchdict, atomwise)
        e_xtb = jax.vmap(self._xtb_energy_single)(
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
        )
        e_mace = mace_out["interaction_energy"]
        return e_xtb + e_mace + self.offset.value

    def _pack_batch(self, batchdict, atomwise):
        """Pad variable-size molecules to fixed shapes for vmapping."""
        ptr = jnp.asarray(batchdict["ptr"])
        ptr_mm = jnp.asarray(batchdict["ptr_mm"])
        n_graph = ptr.shape[0] - 1
        n_qm_each = ptr[1:] - ptr[:-1]
        n_mm_each = ptr_mm[1:] - ptr_mm[:-1]
        max_qm = int(n_qm_each.max())
        max_mm = int(n_mm_each.max())

        def init(shape, dtype):
            return jnp.zeros(shape, dtype=dtype)

        zqm = init((n_graph, max_qm), jnp.int32)
        Rqm = init((n_graph, max_qm, 3), jnp.float64)
        qqm = init((n_graph, max_qm), jnp.float64)
        atomwise_p = init((n_graph, max_qm, self.head_dim), jnp.float64)
        mask_qm = init((n_graph, max_qm), jnp.float32)

        zmm = init((n_graph, max_mm), jnp.int32)
        Rmm = init((n_graph, max_mm, 3), jnp.float64)
        qmm = init((n_graph, max_mm), jnp.float64)
        mask_mm = init((n_graph, max_mm), jnp.float32)

        cell = init((n_graph, 3, 3), jnp.float64)

        def update_slice(base, data, start, length):
            return base.at[:, start : start + length].set(data)

        for i in range(n_graph):
            p1, p2 = int(ptr[i]), int(ptr[i + 1])
            q1, q2 = int(ptr_mm[i]), int(ptr_mm[i + 1])
            nq = p2 - p1
            nm = q2 - q1

            zqm = zqm.at[i, :nq].set(batchdict["z"][p1:p2])
            Rqm = Rqm.at[i, :nq, :].set(batchdict["positions"][p1:p2])
            qqm = qqm.at[i, :nq].set(batchdict["q"][p1:p2])
            atomwise_p = atomwise_p.at[i, :nq, :].set(atomwise[p1:p2])
            mask_qm = mask_qm.at[i, :nq].set(1.0)

            zmm = zmm.at[i, :nm].set(batchdict["z_mm"][q1:q2])
            Rmm = Rmm.at[i, :nm, :].set(batchdict["positions_mm"][q1:q2])
            qmm = qmm.at[i, :nm].set(batchdict["q_mm"][q1:q2])
            mask_mm = mask_mm.at[i, :nm].set(1.0)

            cell = cell.at[i].set(batchdict["cell"][i])

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
        kcn_d3 = param.kcn_d3 * gfactors[2]
        return replace(param, kf=kf, kEN=kEN, kcn_d3=kcn_d3)

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
        shell_mask = jnp.asarray(self.basis.mask_shl[np.asarray(numbers, dtype=int)])
        shell_part = jnp.where(shell_mask[..., None], shell_part, 1.0)

        flat = lambda idx: shell_part[..., idx].reshape(-1)
        atm_to_bas = jnp.asarray(atom_to_bas_indices(mol), dtype=jnp.int32)

        atom_part = atom_part * atom_mask[..., None]

        return replace(
            param,
            arep=param.arep * atom_part[:, 0],
            zeff=param.zeff * atom_part[:, 1],
            gam=param.gam * atom_part[:, 2][atm_to_bas],
            gam3=param.gam3 * atom_part[:, 3],
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
    ) -> jnp.ndarray:
        coords_bohr = anp.asarray(Rqm * A / Bohr)
        mm_coords_bohr = anp.asarray(Rmm * A / Bohr)
        cell_bohr = anp.asarray(cell * A / Bohr)

        charge = int(-jnp.round(jnp.sum(qmm * mask_mm)))
        mol = MolePad(
            anp.asarray(zqm, dtype=anp.int32),
            coords_bohr,
            basis=self.basis,
            verbose=0,
            trace_coords=True,
            charge=charge,
        )

        param_arr = self._apply_global(self.xtb_param, gfactors)
        param_mol = param_arr.to_mol_param(mol)
        param_mol = self._apply_atomwise(mol, param_mol, zqm, atomwise, mask_qm)

        mf = GFN1XTB(mol, param_mol)
        mm_radii = self.mm_radii_table[jnp.asarray(zmm, dtype=jnp.int32)]
        mm_radii = mm_radii * mask_mm
        mf = add_mm_charges(
            mf,
            mm_coords_bohr,
            cell_bohr,
            anp.asarray(qmm * mask_mm),
            anp.asarray(mm_radii),
            unit="Bohr",
            pbcqm=True,
        )
        mf.diis = "qbroyden"
        mf.conv_tol = 1e-7
        mf.diis_damp = 0.6
        energy = mf.kernel()
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


def save_checkpoint(params: dict, path: str):
    with open(path, "wb") as f:
        f.write(jax.serialization.to_bytes(params))


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    return jax.serialization.from_bytes({}, data)


__all__ = ["XTBModel", "energy_loss", "force_loss", "save_checkpoint", "load_checkpoint"]
