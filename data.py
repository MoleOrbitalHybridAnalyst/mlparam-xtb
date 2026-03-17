from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from mace_jax.data.neighborhood import get_neighborhood
from mace_jax.data.utils import AtomicNumberTable, atomic_numbers_to_indices


def pad_batch(batch: dict[str, np.ndarray], n_node: int, n_edge: int, n_mm_node: int, n_graph: int) -> dict[str, np.ndarray]:
    """Pad a collated batch dictionary to fixed capacities."""
    n_nodes_orig = batch["z"].shape[0]
    n_mm_nodes_orig = batch["z_mm"].shape[0]
    n_edges_orig = batch["edge_index"].shape[1]
    n_graphs_orig = batch["cell"].shape[0]

    pad_nodes = n_node - n_nodes_orig
    pad_edges = n_edge - n_edges_orig
    pad_mm_nodes = n_mm_node - n_mm_nodes_orig
    pad_graphs = n_graph - n_graphs_orig

    if pad_nodes < 0 or pad_edges < 0 or pad_mm_nodes < 0 or pad_graphs < 0:
        raise ValueError(f"Batch exceeds padding capacities: nodes {n_nodes_orig}/{n_node}, "
                         f"edges {n_edges_orig}/{n_edge}, mm_nodes {n_mm_nodes_orig}/{n_mm_node}, "
                         f"graphs {n_graphs_orig}/{n_graph}")

    def pad_array(arr, pad_width, axis=0, constant_values=0):
        if pad_width == 0: return arr
        pw = [(0, 0)] * arr.ndim
        pw[axis] = (0, pad_width)
        return np.pad(arr, pw, mode='constant', constant_values=constant_values)

    padded = {}
    padded["z"] = pad_array(batch["z"], pad_nodes)
    padded["q"] = pad_array(batch["q"], pad_nodes)
    padded["positions"] = pad_array(batch["positions"], pad_nodes)
    padded["node_attrs"] = pad_array(batch["node_attrs"], pad_nodes)
    padded["batch"] = pad_array(batch["batch"], pad_nodes, constant_values=n_graphs_orig)

    padded["z_mm"] = pad_array(batch["z_mm"], pad_mm_nodes)
    padded["q_mm"] = pad_array(batch["q_mm"], pad_mm_nodes)
    padded["positions_mm"] = pad_array(batch["positions_mm"], pad_mm_nodes)
    padded["batch_mm"] = pad_array(batch["batch_mm"], pad_mm_nodes, constant_values=n_graphs_orig)

    # Point dummy edges to the first padding node
    dummy_edge_idx = n_nodes_orig if pad_nodes > 0 else 0
    padded["edge_index"] = pad_array(batch["edge_index"], pad_edges, axis=1, constant_values=dummy_edge_idx)
    padded["shifts"] = pad_array(batch["shifts"], pad_edges)
    if pad_edges > 0:
        # Assign non-zero shifts to dummy edges to avoid NaN gradients in MACE norm
        padded["shifts"][n_edges_orig:, :] = 1.0
    padded["unit_shifts"] = pad_array(batch["unit_shifts"], pad_edges)
    if pad_edges > 0:
        padded["unit_shifts"][n_edges_orig:, :] = 1.0

    padded["cell"] = pad_array(batch["cell"], pad_graphs)
    if pad_graphs > 0:
        # Give padding graphs a non-singular cell
        padded["cell"][n_graphs_orig:] = np.eye(3)

    padded_ptr = np.zeros(n_graph + 1, dtype=batch["ptr"].dtype)
    padded_ptr[:n_graphs_orig + 1] = batch["ptr"]
    padded_ptr[n_graphs_orig + 1:] = n_nodes_orig + pad_nodes
    padded["ptr"] = padded_ptr

    padded_ptr_mm = np.zeros(n_graph + 1, dtype=batch["ptr_mm"].dtype)
    padded_ptr_mm[:n_graphs_orig + 1] = batch["ptr_mm"]
    padded_ptr_mm[n_graphs_orig + 1:] = n_mm_nodes_orig + pad_mm_nodes
    padded["ptr_mm"] = padded_ptr_mm

    if "forces" in batch:
        padded["forces"] = pad_array(batch["forces"], pad_nodes)
    if "forces_mm" in batch:
        padded["forces_mm"] = pad_array(batch["forces_mm"], pad_mm_nodes)
    if "energy" in batch:
        padded["energy"] = pad_array(batch["energy"], pad_graphs)

    padded["graph_mask"] = pad_array(np.ones(n_graphs_orig, dtype=bool), pad_graphs, constant_values=False)
    padded["node_mask"] = pad_array(np.ones(n_nodes_orig, dtype=bool), pad_nodes, constant_values=False)
    padded["mm_node_mask"] = pad_array(np.ones(n_mm_nodes_orig, dtype=bool), pad_mm_nodes, constant_values=False)

    return padded


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
        "edge_index": np.asarray(np.concatenate(edge_index, axis=1), dtype=np.int32),
        "shifts": np.asarray(np.concatenate(shifts, axis=0)),
        "unit_shifts": np.asarray(np.concatenate(unit_shifts, axis=0)),
        "positions": np.asarray(np.concatenate(positions, axis=0)),
        "node_attrs": np.asarray(np.concatenate(node_attrs, axis=0)),
        "z": np.asarray(np.concatenate(z, axis=0), dtype=np.int32),
        "q": np.asarray(np.concatenate(q, axis=0)),
        "positions_mm": np.asarray(np.concatenate(positions_mm, axis=0)),
        "z_mm": np.asarray(np.concatenate(z_mm, axis=0), dtype=np.int32),
        "q_mm": np.asarray(np.concatenate(q_mm, axis=0)),
        "batch": np.asarray(np.concatenate(batch_qm, axis=0), dtype=np.int32),
        "batch_mm": np.asarray(np.concatenate(batch_mm, axis=0), dtype=np.int32),
        "ptr": np.asarray(np.asarray(ptr, dtype=np.int32)),
        "ptr_mm": np.asarray(np.asarray(ptr_mm, dtype=np.int32)),
        "cell": np.asarray(np.concatenate(cells, axis=0)),
    }
    if forces:
        result["forces"] = np.asarray(np.concatenate(forces, axis=0))
    if forces_mm:
        result["forces_mm"] = np.asarray(np.concatenate(forces_mm, axis=0))
    if energies:
        result["energy"] = np.asarray(np.concatenate(energies, axis=0))
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
        pad: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self.pad = pad

        if self.pad:
            max_qm_single = max(s.num_nodes for s in dataset.samples)
            max_mm_single = max(s.n_mm for s in dataset.samples)
            max_edges_single = max(s.edge_index.shape[1] for s in dataset.samples)
            
            self.n_node = max_qm_single * batch_size + 1
            self.n_mm_node = max_mm_single * batch_size + 1
            self.n_edge = max_edges_single * batch_size + 1
            self.n_graph = batch_size + 1

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            end = start + self.batch_size
            if end > len(indices) and self.drop_last:
                break
            batch_items = [self.dataset[int(i)] for i in indices[start:end]]
            batch = _collate(batch_items)
            if self.pad:
                batch = pad_batch(batch, self.n_node, self.n_edge, self.n_mm_node, self.n_graph)
            yield {k: jnp.asarray(v) for k, v in batch.items()}


__all__ = ["QMMMData", "QMMMDataset", "DataLoader", "pad_batch"]
