"""
Molecular targets, descriptor baselines, and SMARTS patterns.

Single source of truth for every label used by the probes and by feature
characterisation, so the two cannot silently disagree.

Two deliberate choices, both of which must be stated in any write-up:

1. ``complexity`` reproduces the definition used in the original paper, which is
   **not** a synthetic-accessibility score. It is
   ``1 - min(1, n_aliphatic_rings*0.1 + n_aromatic_rings*0.05 + 0.5)`` — a
   clipped function of two ring counts. It is kept only so new numbers are
   comparable to the published ones, and is named ``complexity_ringproxy`` to
   stop the mislabelling from propagating.
2. RDKit is version-sensitive for QED and descriptors. The version actually used
   is recorded by ``version_info()`` and written into every output table.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------- regression
REGRESSION_TARGETS = [
    "drug_likeness",
    "complexity_ringproxy",
    "lipophilicity",
    "size",
    "polarity",
    "flexibility",
    "molecular_weight",
]

# ------------------------------------------------------------ classification
CLASSIFICATION_SMARTS: Dict[str, str] = {
    "halogen": "[F,Cl,Br,I]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "boron": "[B]",
    # aromaticity handled separately: any aromatic ring
}
CLASSIFICATION_TARGETS = ["halogen", "aromaticity", "urea", "boron"]

ALL_TARGETS = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

# SMARTS used to describe what a feature fires on (feature characterisation)
CHARACTERIZATION_SMARTS: Dict[str, str] = {
    "halogen_F": "[F]",
    "halogen_Cl": "[Cl]",
    "halogen_Br": "[Br]",
    "halogen_I": "[I]",
    "boron": "[B]",
    "boronic_acid_or_ester": "[BX3]([OX2])[OX2]",
    "nitro": "[NX3](=O)=O",
    "nitrile": "[NX1]#[CX2]",
    "aromatic_N": "[n]",
    "aromatic_O": "[o]",
    "aromatic_S": "[s]",
    "pyridine": "n1ccccc1",
    "benzene": "c1ccccc1",
    "phenol": "[OX2H][c]",
    "alcohol": "[OX2H][CX4]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])",
    "ether": "[OD2]([#6])[#6]",
    "amine_primary": "[NX3;H2;!$(NC=O)]",
    "amine_secondary": "[NX3;H1;!$(NC=O)]",
    "amine_tertiary": "[NX3;H0;!$(NC=O);!$(N=*)]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alkyne": "[CX2]#[CX2]",
}

# descriptors compared between high-activating and silent molecules
CHARACTERIZATION_DESCRIPTORS = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "QED",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "NumAliphaticRings",
    "HeavyAtomCount",
    "FractionCSP3",
]


def version_info() -> Dict[str, str]:
    import rdkit

    return {"rdkit": rdkit.__version__}


def _ring_proxy_sa(mol) -> float:
    """The original paper's stand-in for synthetic accessibility. Not an SA score."""
    return min(
        1.0,
        Descriptors.NumAliphaticRings(mol) * 0.1
        + Descriptors.NumAromaticRings(mol) * 0.05
        + 0.5,
    )


def _has_aromatic_ring(mol) -> int:
    return int(any(a.GetIsAromatic() for a in mol.GetAtoms()))


def compute_targets(smiles: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute all 11 targets.

    Returns ``(targets_df, valid_mask)``. Invalid SMILES are excluded via the
    mask rather than filled with zeros — zero-filling silently biases every
    regression, which is a real defect in the original pipeline.
    """
    patterns = {k: Chem.MolFromSmarts(v) for k, v in CLASSIFICATION_SMARTS.items()}
    rows, valid = [], np.zeros(len(smiles), dtype=bool)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            mw = Descriptors.MolWt(mol)
            rec = {
                "drug_likeness": Descriptors.qed(mol),
                "complexity_ringproxy": 1.0 - _ring_proxy_sa(mol),
                "lipophilicity": Descriptors.MolLogP(mol),
                "size": mw / 500.0,
                "polarity": Descriptors.TPSA(mol) / 200.0,
                "flexibility": Descriptors.NumRotatableBonds(mol) / 10.0,
                "molecular_weight": mw,
                "aromaticity": _has_aromatic_ring(mol),
            }
            for name, patt in patterns.items():
                rec[name] = int(mol.HasSubstructMatch(patt))
        except Exception:
            continue
        if not all(np.isfinite(v) for v in rec.values()):
            continue
        rows.append(rec)
        valid[i] = True

    return pd.DataFrame(rows)[ALL_TARGETS], valid


DESCRIPTOR_BASELINE_COLS = [
    "HeavyAtomCount",
    "n_halogen",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "FractionCSP3",
    "n_N",
    "n_O",
    "n_S",
    "n_B",
]


# Baseline columns that *define* a target, making the comparison arithmetic
# rather than informative. E.g. flexibility == NumRotatableBonds/10, so leaving
# NumRotatableBonds in the baseline guarantees R^2 = 1.000 and tells you nothing.
# The probes report the baseline both with and without these, so the degenerate
# case stays visible instead of being silently dropped.
TARGET_LEAKY_BASELINE_COLS: Dict[str, List[str]] = {
    "flexibility": ["NumRotatableBonds"],
    "complexity_ringproxy": ["NumAromaticRings", "NumAliphaticRings", "RingCount"],
    "halogen": ["n_halogen"],
    "boron": ["n_B"],
    "aromaticity": ["NumAromaticRings", "RingCount"],
    # size / molecular_weight are correlated with HeavyAtomCount but not defined
    # by it, and polarity is not defined by HBD/HBA counts, so those comparisons
    # are legitimate and nothing is excluded.
}


def leaky_column_indices(target: str) -> List[int]:
    """Positions in the count block that must be dropped for a fair baseline."""
    return [
        DESCRIPTOR_BASELINE_COLS.index(c)
        for c in TARGET_LEAKY_BASELINE_COLS.get(target, [])
        if c in DESCRIPTOR_BASELINE_COLS
    ]


def compute_descriptor_baseline(
    smiles: List[str], ecfp_bits: int = 1024, radius: int = 2
) -> np.ndarray:
    """Cheap RDKit counts + ECFP4 — the A3 "trivially decodable" baseline.

    A probe on these has no access to the model at all. If it matches the
    embedding probe, the property was simply present in the input.
    """
    feats = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            feats.append(np.zeros(len(DESCRIPTOR_BASELINE_COLS) + ecfp_bits, np.float32))
            continue
        counts = {
            "HeavyAtomCount": mol.GetNumHeavyAtoms(),
            "n_halogen": sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in mol.GetAtoms()),
            "RingCount": rdMolDescriptors.CalcNumRings(mol),
            "NumAromaticRings": Descriptors.NumAromaticRings(mol),
            "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "FractionCSP3": Descriptors.FractionCSP3(mol),
            "n_N": sum(a.GetSymbol() == "N" for a in mol.GetAtoms()),
            "n_O": sum(a.GetSymbol() == "O" for a in mol.GetAtoms()),
            "n_S": sum(a.GetSymbol() == "S" for a in mol.GetAtoms()),
            "n_B": sum(a.GetSymbol() == "B" for a in mol.GetAtoms()),
        }
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=ecfp_bits)
        feats.append(
            np.concatenate(
                [
                    np.array([counts[c] for c in DESCRIPTOR_BASELINE_COLS], np.float32),
                    np.array(fp, np.float32),
                ]
            )
        )
    return np.stack(feats)


def compute_characterization_descriptors(smiles: List[str]) -> pd.DataFrame:
    """Descriptor table used to describe what a feature fires on."""
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({d: np.nan for d in CHARACTERIZATION_DESCRIPTORS})
            continue
        rows.append(
            {
                "MolWt": Descriptors.MolWt(mol),
                "MolLogP": Descriptors.MolLogP(mol),
                "TPSA": Descriptors.TPSA(mol),
                "QED": Descriptors.qed(mol),
                "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
                "NumHDonors": Descriptors.NumHDonors(mol),
                "NumHAcceptors": Descriptors.NumHAcceptors(mol),
                "NumAromaticRings": Descriptors.NumAromaticRings(mol),
                "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
                "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
            }
        )
    return pd.DataFrame(rows)[CHARACTERIZATION_DESCRIPTORS]


def substructure_matrix(smiles: List[str]) -> pd.DataFrame:
    """Binary presence of each characterisation SMARTS."""
    patts = {k: Chem.MolFromSmarts(v) for k, v in CHARACTERIZATION_SMARTS.items()}
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({k: 0 for k in patts})
            continue
        rows.append({k: int(mol.HasSubstructMatch(p)) for k, p in patts.items()})
    return pd.DataFrame(rows)
