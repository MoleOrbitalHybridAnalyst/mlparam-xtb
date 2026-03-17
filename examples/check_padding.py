from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from pyscfad.ml.gto import make_basis_array
from pyscfad.xtb import basis as xtb_basis
from pyscfad.ml.xtb.param import make_param_array

from mace_jax.data.utils import AtomicNumberTable
from mace_jax.modules import ScaleShiftMACE, RealAgnosticInteractionBlock
from mlparam_xtb.data import QMMMData, _collate
from mlparam_xtb.models import XTBModel
from mlparam_xtb.utils import scalar_node_feature_indices



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


def main():
    # Basis / parameters
    bfile = xtb_basis.get_basis_filename()
    basis = make_basis_array(bfile, max_number=8)
    param = make_param_array(basis, max_number=8)
    param.dipgam = jnp.array(param.gam)
    param.quadgam = jnp.array(param.gam)

    # Atomic number table from QM atoms
    z_table = AtomicNumberTable([1, 8])

    # Build two samples and collate
    samples = [make_sample(z_table), make_sample_two(z_table)]
    batch = _collate(samples)
    batch = jax.device_put(batch)

    max_qm = max(s.num_nodes for s in samples)
    max_mm = max(s.n_mm for s in samples)

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
        hidden_irreps="16x0e+16x1o",
        MLP_irreps="16x0e",
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
        scf_verbose=4,
        preserve_sign=False,
    )

    # Zero out the energy prediction heads in mace to predict zero energies
    for readout in model.mace.readouts:
        state = nnx.state(readout, nnx.Param)
        zero_state = jax.tree_util.tree_map(jnp.zeros_like, state)
        nnx.update(readout, zero_state)
    
    model.mace.atomic_energies_fn.atomic_energies[...] = \
        model.mace.atomic_energies_fn.atomic_energies * 0.
    model.offset[...] = model.offset * 0.

    from pyscfad.gto.mole_lite import MoleLite
    from pyscfad.xtb import GFN1XTB
    from pyscfad.xtb.qmmm_pbc.itrf import add_mm_charges
    from pyscfad.xtb.param import GFN1Param
    from pyscfad.xtb.util import load_unique_element_params
    from mlparam_xtb.constants import A, Bohr, hartree, eV

    e_exact = []
    g_qm_exact = []
    g_mm_exact = []

    for i, s in enumerate(samples):
        def energy_fn(pos_qm, pos_mm):
            coords_bohr = jnp.asarray(pos_qm * A / Bohr)
            mm_coords_bohr = jnp.asarray(pos_mm * A / Bohr)
            cell_bohr = jnp.asarray(s.cell * A / Bohr)
            
            mol = MoleLite(
                numbers=tuple(int(z) for z in s.z),
                coords=coords_bohr,
                basis=bfile,
                verbose=model.scf_verbose,
                trace_coords=True,
                charge=-jnp.round(jnp.sum(s.q_mm)).astype(int),
            )
            
            mf = GFN1XTB(mol, GFN1Param())
            mf.param.dipgam = load_unique_element_params(mol, GFN1Param(), "gam", broadcast="atom")
            mf.param.quadgam = load_unique_element_params(mol, GFN1Param(), "gam", broadcast="atom")
            kdiag = jnp.diag(mf.param.k_shlpr)
            mf.param.k_shlpr = 0.5 * (kdiag[:, None] + kdiag[None, :])
            mm_radii = model.mm_radii_table[jnp.asarray(s.z_mm, dtype=jnp.int32)]
            
            mf = add_mm_charges(
                mf,
                mm_coords_bohr,
                cell_bohr,
                jnp.asarray(s.q_mm),
                jnp.asarray(mm_radii),
                max_mm_nbr=model.max_mm_nbr,
                mm_ew_rcut=model.mm_ew_rcut,
                mm_ew_mesh=model.mm_ew_mesh,
                qm_ew_mesh=model.qm_ew_mesh,
                ew_precision=model.ew_precision,
                unit="Bohr",
                pbcqm=True,
            )
            mf.diis = "qbroyden"
            mf.conv_tol = model.scf_conv_tol
            mf.diis_damp = 0.6
            energy = mf.kernel()
            return jnp.asarray(energy) * hartree / eV

        e, (g_qm, g_mm) = jax.jit(jax.value_and_grad(energy_fn, argnums=(0, 1)))(s.positions, s.positions_mm)
        e_exact.append(e)
        g_qm_exact.append(g_qm)
        g_mm_exact.append(g_mm)

    @nnx.jit
    def batched_energy_and_grad(model, batch):
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
        return e_pred, fqm, fmm

    e_pred_batch, g_qm_batch, g_mm_batch = batched_energy_and_grad(model, batch)

    print("\nEnergies:")
    print("Batched:", e_pred_batch[:2])
    print("Exact:  ", jnp.array(e_exact))

    print("\nQM Gradients max diff:")
    for i in range(2):
        mask = batch["batch"] == i
        g_qm_b = g_qm_batch[mask]
        diff = jnp.abs(g_qm_b - g_qm_exact[i]).max()
        print(f"  Sample {i}: {diff:.6e}")

    print("\nMM Gradients max diff:")
    for i in range(2):
        mask = batch["batch_mm"] == i
        g_mm_b = g_mm_batch[mask]
        diff = jnp.abs(g_mm_b - g_mm_exact[i]).max()
        print(f"  Sample {i}: {diff:.6e}")


if __name__ == "__main__":
    main()
