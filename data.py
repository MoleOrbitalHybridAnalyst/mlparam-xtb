from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from mace_jax.data.neighborhood import get_neighborhood
from mace_jax.data.utils import AtomicNumberTable, atomic_numbers_to_indices


@dataclass
class QMMMData:
    """Single QM/MM configuration stored as numpy arrays."""

    edge_index: np.ndarray  # [2, n_edges]
    node_attrs: np.ndarray  # [n_qm, n_node_feats]
    positions: np.ndarray  # [n_qm, 3]  (Angstrom)
    shifts: np.ndarray  # [n_edges, 3]
    unit_shifts: np.ndarray  # [n_edges, 3]
    cell: np.ndarray  # [3, 3]
    z: np.ndarray  # QM atomic numbers [n_qm]
    q: np.ndarray  # QM partial charges [n_qm]
    positions_mm: np.ndarray  # [n_mm, 3]
    z_mm: np.ndarray  # MM atomic numbers [n_mm]
    q_mm: np.ndarray  # MM partial charges [n_mm]
    forces: Optional[np.ndarray] = None  # [n_qm, 3]
    energy: Optional[np.ndarray] = None  # scalar
    forces_mm: Optional[np.ndarray] = None  # [n_mm, 3]

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_mm(self) -> int:
        return int(self.positions_mm.shape[0])

    @classmethod
    def from_raw(
        cls,
        zqm: np.ndarray,
        Rqm: np.ndarray,
        zmm: np.ndarray,
        Rmm: np.ndarray,
        a: np.ndarray,
        E: Optional[np.ndarray],
        Fqm: Optional[np.ndarray],
        Fmm: Optional[np.ndarray],
        qqm: np.ndarray,
        qmm: np.ndarray,
        z_table: AtomicNumberTable,
        cutoff: float,
    ) -> "QMMMData":
        """Build a QMMMData object from raw arrays in Angstrom / eV units."""

        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=Rqm,
            cutoff=cutoff,
            pbc=(True, True, True),
            cell=a,
        )
        indices = atomic_numbers_to_indices(zqm, z_table)
        node_attrs = jnp.asarray(
            jnp.eye(len(z_table), dtype=jnp.float64)[indices]
        ).astype(np.float64)

        return cls(
            edge_index=np.asarray(edge_index, dtype=np.int32),
            node_attrs=np.asarray(node_attrs, dtype=np.float64),
            positions=np.asarray(Rqm, dtype=np.float64),
            shifts=np.asarray(shifts, dtype=np.float64),
            unit_shifts=np.asarray(unit_shifts, dtype=np.float64),
            cell=np.asarray(cell, dtype=np.float64),
            z=np.asarray(zqm, dtype=np.int32),
            q=np.asarray(qqm, dtype=np.float64),
            positions_mm=np.asarray(Rmm, dtype=np.float64),
            z_mm=np.asarray(zmm, dtype=np.int32),
            q_mm=np.asarray(qmm, dtype=np.float64),
            energy=None if E is None else np.asarray(E, dtype=np.float64),
            forces=None if Fqm is None else np.asarray(Fqm, dtype=np.float64),
            forces_mm=None if Fmm is None else np.asarray(Fmm, dtype=np.float64),
        )


class QMMMDataset:
    """In-memory dataset backed by one or more NPZ archives."""

    def __init__(
        self,
        npz_files: Sequence[str],
        dataslices: Sequence[Iterable[int] | slice],
        z_table: AtomicNumberTable,
        cutoff: float,
    ):
        """
        Args:
            npz_files: paths to .npz files (units: Bohr for coords, Hartree for energy)
            dataslices: iterable of frame indices for each npz file
            z_table: AtomicNumberTable used to build one-hot node features
            cutoff: neighbor cutoff (Angstrom)
        """
        from constants import Bohr, A, eV, hartree  # pylint: disable=import-error
        import numpy as np

        self.samples: list[QMMMData] = []
        if isinstance(npz_files, str):
            npz_files = [npz_files]

        for file_path, dataslice in zip(npz_files, dataslices):
            data = np.load(file_path)
            z = data["z"]  # (natom,)
            R = data["R"] * Bohr / A  # -> Angstrom
            F = data["F"] * (hartree / Bohr) / (eV / A)
            E = data["E"] * hartree / eV
            a = data["a"] * Bohr / A
            qm_indexes = data["qm_indexes"]
            q = data["q"]

            mm_indexes = [i for i in range(z.shape[0]) if i not in qm_indexes]

            ndata = R.shape[0]
            for i in np.arange(ndata)[dataslice]:
                zqm = z[qm_indexes]
                zmm = z[mm_indexes]
                Rqm = R[i, qm_indexes, :]
                Rmm = R[i, mm_indexes, :]
                Fqm = F[i, qm_indexes, :]
                Fmm = F[i, mm_indexes, :]
                qqm = q[qm_indexes]
                qmm = q[mm_indexes]

                sample = QMMMData.from_raw(
                    zqm=zqm,
                    Rqm=Rqm,
                    zmm=zmm,
                    Rmm=Rmm,
                    a=a,
                    E=E[i],
                    Fqm=Fqm,
                    Fmm=Fmm,
                    qqm=qqm,
                    qmm=qmm,
                    z_table=z_table,
                    cutoff=cutoff,
                )
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> QMMMData:
        return self.samples[idx]


def _collate(batch: List[QMMMData]) -> dict[str, jnp.ndarray]:
    """Concatenate a list of QMMMData objects into a single batch dict."""
    edge_index = []
    shifts = []
    unit_shifts = []
    positions = []
    node_attrs = []
    z = []
    q = []
    positions_mm = []
    z_mm = []
    q_mm = []
    ptr = [0]
    ptr_mm = [0]
    batch_qm = []
    batch_mm = []
    cells = []
    forces = []
    forces_mm = []
    energies = []

    for i, data in enumerate(batch):
        offset = ptr[-1]
        positions.append(data.positions)
        node_attrs.append(data.node_attrs)
        z.append(data.z)
        q.append(data.q)
        cells.append(data.cell[None, ...])
        if data.forces is not None:
            forces.append(data.forces)
        if data.energy is not None:
            energies.append(np.asarray(data.energy)[None])

        edge_index.append(data.edge_index + offset)
        shifts.append(data.shifts)
        unit_shifts.append(data.unit_shifts)
        ptr.append(offset + data.num_nodes)
        batch_qm.append(np.full((data.num_nodes,), i, dtype=np.int32))

        offset_mm = ptr_mm[-1]
        positions_mm.append(data.positions_mm)
        z_mm.append(data.z_mm)
        q_mm.append(data.q_mm)
        ptr_mm.append(offset_mm + data.n_mm)
        batch_mm.append(np.full((data.n_mm,), i, dtype=np.int32))
        if data.forces_mm is not None:
            forces_mm.append(data.forces_mm)

    result = {
        "edge_index": jnp.asarray(np.concatenate(edge_index, axis=1), dtype=jnp.int32),
        "shifts": jnp.asarray(np.concatenate(shifts, axis=0)),
        "unit_shifts": jnp.asarray(np.concatenate(unit_shifts, axis=0)),
        "positions": jnp.asarray(np.concatenate(positions, axis=0)),
        "node_attrs": jnp.asarray(np.concatenate(node_attrs, axis=0)),
        "z": jnp.asarray(np.concatenate(z, axis=0), dtype=jnp.int32),
        "q": jnp.asarray(np.concatenate(q, axis=0)),
        "positions_mm": jnp.asarray(np.concatenate(positions_mm, axis=0)),
        "z_mm": jnp.asarray(np.concatenate(z_mm, axis=0), dtype=jnp.int32),
        "q_mm": jnp.asarray(np.concatenate(q_mm, axis=0)),
        "batch": jnp.asarray(np.concatenate(batch_qm, axis=0), dtype=jnp.int32),
        "batch_mm": jnp.asarray(np.concatenate(batch_mm, axis=0), dtype=jnp.int32),
        "ptr": jnp.asarray(np.asarray(ptr, dtype=np.int32)),
        "ptr_mm": jnp.asarray(np.asarray(ptr_mm, dtype=np.int32)),
        "cell": jnp.asarray(np.concatenate(cells, axis=0)),
    }
    if forces:
        result["forces"] = jnp.asarray(np.concatenate(forces, axis=0))
    if forces_mm:
        result["forces_mm"] = jnp.asarray(np.concatenate(forces_mm, axis=0))
    if energies:
        result["energy"] = jnp.asarray(np.concatenate(energies, axis=0))
    return result


class DataLoader:
    """Minimal DataLoader analogue that yields JAX-ready batch dictionaries."""

    def __init__(
        self,
        dataset: QMMMDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if end > len(indices) and self.drop_last:
                break
            batch_items = [self.dataset[int(i)] for i in indices[start:end]]
            yield _collate(batch_items)


__all__ = ["QMMMData", "QMMMDataset", "DataLoader"]
