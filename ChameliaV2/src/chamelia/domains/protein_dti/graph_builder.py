"""Graph payload helpers for the protein DTI domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
ATOM_VOCAB = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
    "P",
    "Na",
    "K",
    "Ca",
    "Mg",
    "Fe",
    "Zn",
    "Cu",
    "OTHER",
]
ATOM_TO_IDX = {symbol: idx for idx, symbol in enumerate(ATOM_VOCAB)}
DISTANCE_CUTOFF = 10.0


@dataclass
class GraphRecord:
    """Torch-native graph payload saved in ``.pt`` files."""

    identifier: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize the graph to a plain ``torch.save``-friendly dictionary."""
        return {
            "identifier": self.identifier,
            "x": self.x.detach().cpu(),
            "edge_index": self.edge_index.detach().cpu(),
            "edge_attr": self.edge_attr.detach().cpu(),
            "metadata": dict(self.metadata),
        }


def coerce_graph_record(graph: Any, *, identifier_fallback: str = "") -> GraphRecord:
    """Normalize a saved graph payload into a ``GraphRecord``."""
    if isinstance(graph, GraphRecord):
        return graph

    if isinstance(graph, dict):
        identifier = str(graph.get("identifier") or identifier_fallback)
        x = torch.as_tensor(graph["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
        edge_attr = torch.as_tensor(graph["edge_attr"], dtype=torch.float32)
        metadata = dict(graph.get("metadata", {}))
        return GraphRecord(
            identifier=identifier,
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            metadata=metadata,
        )

    if all(hasattr(graph, field_name) for field_name in ("x", "edge_index", "edge_attr")):
        identifier = str(
            getattr(graph, "identifier", None)
            or getattr(graph, "uniprot_id", None)
            or getattr(graph, "chembl_id", None)
            or identifier_fallback
        )
        metadata = {
            key: value
            for key, value in vars(graph).items()
            if key not in {"x", "edge_index", "edge_attr"}
        }
        return GraphRecord(
            identifier=identifier,
            x=torch.as_tensor(graph.x, dtype=torch.float32),
            edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(graph.edge_attr, dtype=torch.float32),
            metadata=metadata,
        )

    raise TypeError(f"Unsupported graph payload type: {type(graph)!r}")


def save_graph_record(graph: GraphRecord, path: str | Path) -> None:
    """Persist a graph record as a plain dictionary."""
    torch.save(graph.to_payload(), Path(path))


def _get_cb_coords(residue: Any) -> np.ndarray | None:
    if residue.resname == "GLY":
        atom = residue.get("CA")
    else:
        atom = residue.get("CB")
    if atom is None:
        atom = residue.get("CA")
    if atom is None:
        return None
    return atom.get_vector().get_array()


def _residue_to_aa(residue: Any) -> str:
    from Bio.Data.IUPACData import protein_letters_3to1  # type: ignore[import-untyped]

    three_letter = residue.resname.capitalize()
    return protein_letters_3to1.get(three_letter, "X")


def build_protein_graph(
    structure_path: str | Path,
    uniprot_id: str,
    *,
    distance_cutoff: float = DISTANCE_CUTOFF,
) -> GraphRecord | None:
    """Build a residue graph from an mmCIF structure."""
    from Bio.PDB import MMCIFParser  # type: ignore[import-untyped]

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(uniprot_id, str(structure_path))
    model = structure[0]
    residues: list[Any] = []
    coords: list[np.ndarray] = []

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            cb_coord = _get_cb_coords(residue)
            if cb_coord is None:
                continue
            residues.append(residue)
            coords.append(cb_coord)

    if len(residues) < 2:
        return None

    coord_array = np.asarray(coords, dtype=np.float32)
    num_nodes = coord_array.shape[0]
    node_features = np.zeros((num_nodes, len(AA_VOCAB)), dtype=np.float32)
    for index, residue in enumerate(residues):
        aa = _residue_to_aa(residue)
        node_features[index, AA_TO_IDX.get(aa, AA_TO_IDX["X"])] = 1.0

    diff = coord_array[:, None, :] - coord_array[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    src, dst = np.where((dist_matrix < distance_cutoff) & (dist_matrix > 0.0))
    if len(src) == 0:
        return None

    distances = dist_matrix[src, dst]
    edge_vecs = diff[src, dst] / distances[:, None].clip(min=1.0e-6)
    rbf_centers = np.linspace(0.0, distance_cutoff, 16)
    rbf_sigma = 1.0
    rbf = np.exp(-0.5 * ((distances[:, None] - rbf_centers[None, :]) / rbf_sigma) ** 2)

    edge_attr = np.concatenate([rbf, edge_vecs], axis=-1)
    return GraphRecord(
        identifier=uniprot_id,
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        metadata={"structure_path": str(structure_path)},
    )


def build_drug_graph(smiles: str, chembl_id: str) -> GraphRecord | None:
    """Build an atom/bond graph from a SMILES string."""
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem  # type: ignore[import-untyped]

    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.Compute2DCoords(mol)
    mol = Chem.RemoveHs(mol)
    conformer = mol.GetConformer() if mol.GetNumConformers() > 0 else None

    node_features: list[list[float]] = []
    positions: list[list[float]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        symbol_index = ATOM_TO_IDX.get(symbol, ATOM_TO_IDX["OTHER"])
        one_hot = [0.0] * len(ATOM_VOCAB)
        one_hot[symbol_index] = 1.0
        node_features.append(
            one_hot
            + [
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing()),
                float(atom.GetTotalDegree()) / 6.0,
                float(atom.GetTotalNumHs()) / 4.0,
            ]
        )
        if conformer is None:
            positions.append([0.0, 0.0, 0.0])
        else:
            pos = conformer.GetAtomPosition(atom.GetIdx())
            positions.append([float(pos.x), float(pos.y), float(pos.z)])

    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_index = bond_types.get(bond.GetBondType(), 0)
        bond_one_hot = [0.0] * len(bond_types)
        bond_one_hot[bond_index] = 1.0
        distance = float(torch.dist(pos_tensor[begin], pos_tensor[end]).item())
        features = bond_one_hot + [distance, float(bond.IsInRing())]
        for src_idx, dst_idx in ((begin, end), (end, begin)):
            edge_src.append(src_idx)
            edge_dst.append(dst_idx)
            edge_attr.append(features)

    if not edge_src:
        return None

    return GraphRecord(
        identifier=chembl_id,
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        metadata={"smiles": smiles},
    )
