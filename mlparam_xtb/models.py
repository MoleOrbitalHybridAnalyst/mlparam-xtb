from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random
from pyscfad.ml.gto import MolePad
from pyscfad.ml.xtb.param import GFN1ParamArray, GFN1MoleParam
from pyscfad.ml.xtb.xtb_pad import GFN1XTB
from pyscfad.xtb.data.radii import COV_D3
from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges
from pyscfad.xtb.util import atom_to_bas_indices

from mace_jax.modules.models import ScaleShiftMACE

from .constants import A, Bohr, hartree, eV  # pylint: disable=import-error
from .utils import scalar_node_feature_indices

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
        scf_diis_damp: float = 0.6,
        scf_conv_tol: float = 1e-6,
        scf_max_cycle: int = 50,
        scf_verbose: int = 0,
        scf_sigma: float | None = None,
        max_mm_nbr: int = 512,
        mm_ew_rcut: float = 18.0,
        mm_ew_mesh: Sequence[int] | None = None,
        qm_ew_mesh: Sequence[int] = (40, 40, 40),
        n_decoder_layer: int = 1,
        scf_init_guess: str = "atomic_charges",
    ) -> None:
        if scf_init_guess not in ("pyscfad", "atomic_charges", "from_data"):
            raise ValueError(
                f"scf_init_guess must be one of 'pyscfad', 'atomic_charges', 'from_data'; got {scf_init_guess!r}"
            )
        self.scf_init_guess = scf_init_guess
        self.mace = mace_model
        self.xtb_param = nnx.data(xtb_param)
        self.basis = nnx.data(basis)
        self.preserve_sign = preserve_sign
        self.ew_precision = ew_precision
        self.scf_diis_damp = scf_diis_damp
        self.scf_conv_tol = scf_conv_tol
        self.scf_max_cycle = scf_max_cycle
        self.scf_verbose = scf_verbose
        self.scf_sigma = scf_sigma
        self.max_mm_nbr = max_mm_nbr
        self.mm_ew_rcut = mm_ew_rcut
        self.qm_ew_mesh = qm_ew_mesh


        if max_qm is None or max_mm is None:
            raise ValueError("max_qm and max_mm must be provided for static padding.")
        self.max_qm = int(max_qm)
        self.max_mm = int(max_mm)

        if mm_ew_mesh is None:
            raise ValueError("mm_ew_mesh must be provided for static padding."
                             "One grid per one angstrom is recommended.")
        self.mm_ew_mesh = mm_ew_mesh

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

    def __call__(self, batchdict: dict):
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
        cuint_plan = batchdict.get("cuint_plan", None)
        e_xtb = jax.vmap(
            self._xtb_energy_single,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
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
            packed["shell_charges"],
            packed["atom_dips"],
            packed["atom_quads"],
            gfactors,
            xtb_param,
            cuint_plan,
        )

        # Pad e_xtb back to match the total number of graphs in the batch
        ptr = jnp.asarray(batchdict["ptr"])
        n_graph = ptr.shape[0] - 1
        pad_amount = n_graph - e_xtb.shape[0]
        if pad_amount > 0:
            e_xtb = jnp.pad(e_xtb, ((0, pad_amount),))
        e_mace = mace_out["interaction_energy"]
        return e_xtb + e_mace + self.offset.value

    def _pack_batch(self, batchdict, atomwise):
        """Pad variable-size molecules to fixed shapes for vmapping."""
        ptr = jnp.asarray(batchdict["ptr"])
        ptr_mm = jnp.asarray(batchdict["ptr_mm"])
        n_graph = ptr.shape[0] - 1
        # Skip the last graph, which is always the Jraph padding graph
        # This prevents xTB from running expensive SCF operations on dummy atoms
        n_real_graph = max(n_graph - 1, 1)
        max_qm = self.max_qm
        max_mm = self.max_mm

        idx_q = jnp.arange(max_qm)
        idx_m = jnp.arange(max_mm)

        n_nodes = batchdict["z"].shape[0]
        if self.scf_init_guess == "from_data":
            if "shell_charges" not in batchdict:
                raise KeyError(
                    "scf_init_guess='from_data' requires 'shell_charges' in the batch dict"
                )
            shell_charges_full = batchdict["shell_charges"]
            atom_dips_full = batchdict["atom_dips"]
            atom_quads_full = batchdict["atom_quads"]
        else:
            shell_charges_full = jnp.zeros((n_nodes, self.max_shells), dtype=jnp.float64)
            atom_dips_full = jnp.zeros((n_nodes, 3), dtype=jnp.float64)
            atom_quads_full = jnp.zeros((n_nodes, 3, 3), dtype=jnp.float64)

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
            sc = jnp.where(mask[:, None], shell_charges_full[sel], 0.0)
            ad = jnp.where(mask[:, None], atom_dips_full[sel], 0.0)
            aq = jnp.where(mask[:, None, None], atom_quads_full[sel], 0.0)
            maskf = mask.astype(jnp.float64)
            return z, R, q, aw, sc, ad, aq, maskf

        def gather_m(i):
            q1 = ptr_mm[i]
            q2 = ptr_mm[i + 1]
            nm = q2 - q1
            sel = jnp.clip(q1 + idx_m, 0, batchdict["z_mm"].shape[0] - 1)
            mask = idx_m < nm
            z = jnp.where(mask, batchdict["z_mm"][sel], 0)
            R = jnp.where(mask[:, None], batchdict["positions_mm"][sel], 0.)
            q = jnp.where(mask, batchdict["q_mm"][sel], 0.0)
            maskf = mask.astype(jnp.float64)
            return z, R, q, maskf

        zqm, Rqm, qqm, atomwise_p, sc_p, ad_p, aq_p, mask_qm = jax.vmap(gather_q)(jnp.arange(n_real_graph))
        zmm, Rmm, qmm, mask_mm = jax.vmap(gather_m)(jnp.arange(n_real_graph))
        cell = batchdict["cell"][:n_real_graph]

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
            "shell_charges": sc_p,
            "atom_dips": ad_p,
            "atom_quads": aq_p,
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

    def _build_q0(
        self,
        mol,
        param_mol,
        qqm: jnp.ndarray,
        mask_qm: jnp.ndarray,
        shell_charges: jnp.ndarray,
        atom_dips: jnp.ndarray,
        atom_quads: jnp.ndarray,
    ):
        """Construct the SCF initial-guess vector according to ``self.scf_init_guess``.

        Returns ``None`` for the ``pyscfad`` mode (caller passes no ``q0`` to
        ``mf.kernel``), or a packed 1-D array containing the shell-charge block
        followed by optional dipole/quadrupole blocks.
        """
        if self.scf_init_guess == "pyscfad":
            return None

        atm_to_bas = jnp.asarray(atom_to_bas_indices(mol))
        nbas_per_atom = jnp.bincount(atm_to_bas, length=mol.natm)
        safe_nbas = jnp.where(nbas_per_atom > 0, nbas_per_atom, 1)

        if self.scf_init_guess == "atomic_charges":
            q0_shell = (qqm * mask_qm)[atm_to_bas] / safe_nbas[atm_to_bas]
            dip_block = jnp.zeros(mol.natm * 3, dtype=q0_shell.dtype)
            quad_block = jnp.zeros(mol.natm * 9, dtype=q0_shell.dtype)
        else:  # "from_data"
            first_idx = jnp.concatenate(
                [jnp.zeros(1, dtype=nbas_per_atom.dtype), jnp.cumsum(nbas_per_atom)[:-1]]
            )
            shell_in_atom = jnp.arange(atm_to_bas.shape[0]) - first_idx[atm_to_bas]
            q0_shell = shell_charges[atm_to_bas, shell_in_atom] * mask_qm[atm_to_bas]
            dip_block = (atom_dips * mask_qm[:, None]).reshape(-1)
            quad_block = (atom_quads * mask_qm[:, None, None]).reshape(-1)

        q0_parts = [q0_shell]
        if param_mol.dipgam is not None:
            q0_parts.append(dip_block)
        if param_mol.quadgam is not None:
            q0_parts.append(quad_block)
        return jnp.concatenate(q0_parts)

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
        shell_charges: jnp.ndarray,
        atom_dips: jnp.ndarray,
        atom_quads: jnp.ndarray,
        gfactors: jnp.ndarray,
        xtb_param: GFN1ParamArray,
        cuint_plan=None,
    ) -> jnp.ndarray:
        coords_bohr = jnp.asarray(Rqm * A / Bohr)
        mm_coords_bohr = jnp.asarray(Rmm * A / Bohr)
        cell_bohr = jnp.asarray(cell * A / Bohr)

        mol = MolePad(
            jnp.asarray(zqm, dtype=jnp.int32),
            coords_bohr,
            basis=self.basis,
            verbose=self.scf_verbose,
            trace_coords=True,
            charge=jnp.round(jnp.sum(qqm * mask_qm)).astype(int),
            cuint_plan=cuint_plan,
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
        mf.max_cycle = self.scf_max_cycle
        mf.diis_damp = self.scf_diis_damp
        mf.sigma = self.scf_sigma

        q0 = self._build_q0(mol, param_mol, qqm, mask_qm,
                            shell_charges, atom_dips, atom_quads)
        energy = mf.kernel() if q0 is None else mf.kernel(q0=q0)
        return jnp.asarray(energy) * hartree / eV

    def _xtb_charges_single(
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
        shell_charges: jnp.ndarray,
        atom_dips: jnp.ndarray,
        atom_quads: jnp.ndarray,
        gfactors: jnp.ndarray,
        xtb_param: GFN1ParamArray,
        cuint_plan=None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run SCF (using the model's current ``scf_init_guess``) and return
        converged per-atom shell charges, dipoles, and quadrupoles in the
        layout expected by the ``from_data`` init-guess branch."""
        coords_bohr = jnp.asarray(Rqm * A / Bohr)
        mm_coords_bohr = jnp.asarray(Rmm * A / Bohr)
        cell_bohr = jnp.asarray(cell * A / Bohr)

        mol = MolePad(
            jnp.asarray(zqm, dtype=jnp.int32),
            coords_bohr,
            basis=self.basis,
            verbose=self.scf_verbose,
            trace_coords=True,
            charge=jnp.round(jnp.sum(qqm * mask_qm)).astype(int),
            cuint_plan=cuint_plan,
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
        mf.max_cycle = self.scf_max_cycle
        mf.diis_damp = self.scf_diis_damp
        mf.sigma = self.scf_sigma

        q0 = self._build_q0(mol, param_mol, qqm, mask_qm,
                            shell_charges, atom_dips, atom_quads)
        if q0 is None:
            mf.kernel()
        else:
            mf.kernel(q0=q0)
        dm = mf.make_rdm1()
        shell_flat = mf.shell_charges(dm=dm)

        atm_to_bas = jnp.asarray(atom_to_bas_indices(mol))
        nbas_per_atom = jnp.bincount(atm_to_bas, length=mol.natm)
        first_idx = jnp.concatenate(
            [jnp.zeros(1, dtype=nbas_per_atom.dtype), jnp.cumsum(nbas_per_atom)[:-1]]
        )
        shell_in_atom = jnp.arange(atm_to_bas.shape[0]) - first_idx[atm_to_bas]
        shell_full = jnp.zeros((mol.natm, self.max_shells), dtype=shell_flat.dtype)
        shell_full = shell_full.at[atm_to_bas, shell_in_atom].set(shell_flat)
        shell_full = shell_full * mask_qm[:, None]

        if param_mol.dipgam is not None:
            dips = mf.get_qm_dipoles(dm) * mask_qm[:, None]
        else:
            dips = jnp.zeros((mol.natm, 3), dtype=shell_flat.dtype)
        if param_mol.quadgam is not None:
            quads = mf.get_qm_quadrupoles(dm) * mask_qm[:, None, None]
        else:
            quads = jnp.zeros((mol.natm, 3, 3), dtype=shell_flat.dtype)

        return shell_full, dips, quads

    def precompute_charges(self, batchdict: dict):
        """Run MACE + xTB SCF and return per-graph converged
        (shell_charges, atom_dips, atom_quads).

        Shapes (per real graph in the batch):
          shell_charges: (max_qm, max_shells)
          atom_dips:     (max_qm, 3)
          atom_quads:    (max_qm, 3, 3)
        Padded atoms are zeroed.
        """
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
        k_shlpr_diag = jnp.diagonal(base_param.k_shlpr) * k_shlpr_factors
        k_shlpr = 0.5 * (k_shlpr_diag[:, None] + k_shlpr_diag[None, :])
        xtb_param = replace(base_param, k_shlpr=k_shlpr)

        packed = self._pack_batch(batchdict, atomwise)
        cuint_plan = batchdict.get("cuint_plan", None)
        sc, ad, aq = jax.vmap(
            self._xtb_charges_single,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None),
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
            packed["shell_charges"],
            packed["atom_dips"],
            packed["atom_quads"],
            gfactors,
            xtb_param,
            cuint_plan,
        )
        return sc, ad, aq


__all__ = [
    "XTBModel",
]
