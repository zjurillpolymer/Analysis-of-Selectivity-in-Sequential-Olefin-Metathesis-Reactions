#!/usr/bin/env python3
"""Build and draw the two corrected B6 local q1 products with RDKit."""
from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import build_q2_branch_structures as metathesis  # noqa: E402
import build_b6_q1_review as review  # noqa: E402
import build_b6_q1_md_structure as q1_builder  # noqa: E402

OUT = ROOT / "results/b6_local_q1_arm1_competition_20260923"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / "gromacs_dcm_server_bundle/structures/B6_p_C6.pdb"

def source_serial(mol, vinyl, mapping):
    _, retained = metathesis.terminal_and_retained_indices(mol, mapping[vinyl])
    return mol.GetAtomWithIdx(retained).GetIntProp("source_q1_serial")

def bond_for_serials(mol, left_serial, right_serial):
    left = metathesis.atom_index_by_source_serial(mol, left_serial)
    right = metathesis.atom_index_by_source_serial(mol, right_serial)
    bond = mol.GetBondBetweenAtoms(left, right)
    if bond is None:
        raise ValueError(f"missing bond for source serials {left_serial},{right_serial}")
    return bond.GetIdx()

mapping0 = review.read_mapping(ROOT)
substrate = metathesis.load_rdkit_pdb(SOURCE)

# q0 is fixed to V1-V12; its product is the common starting template.
q0_serials = (source_serial(substrate, 1, mapping0), source_serial(substrate, 12, mapping0))
q0 = metathesis.edit_second_metathesis(substrate, mapping0, (1, 12))
present_serials = {
    atom.GetIntProp("source_q1_serial")
    for atom in q0.GetAtoms()
    if atom.HasProp("source_q1_serial")
}
mapping_q0 = {
    vinyl: tuple(serial for serial in serials if serial in present_serials)
    for vinyl, serials in mapping0.items()
    if vinyl not in (1, 12)
}
q0_bond = bond_for_serials(q0, *q0_serials)

AllChem.Compute2DCoords(q0)
Draw.MolToFile(
    q0, str(OUT / "B6_q0_A1_A6_cis_2D.png"), size=(1500, 1100),
    kekulize=True, legend="q0: A1–A6 = V1–V12 (cis)",
)

for label, pair in (("reuse_A1_A6", (2, 11)), ("one_arm_A1_A2", (2, 3))):
    product = metathesis.edit_second_metathesis(q0, mapping_q0, pair)
    AllChem.Compute2DCoords(product)
    q1_serials = (source_serial(q0, pair[0], mapping_q0), source_serial(q0, pair[1], mapping_q0))
    q0_idx = bond_for_serials(product, *q0_serials)
    q1_idx = bond_for_serials(product, *q1_serials)
    # Render a clean molecular graph with both reaction bonds highlighted.
    Draw.MolToFile(
        product, str(OUT / f"B6_q1_{label}_cis_2D.png"), size=(1800, 1200),
        kekulize=True, highlightBonds=[q0_idx, q1_idx],
        legend=f"q0 A1–A6 (V1–V12) + q1 {pair[0]}–{pair[1]} (cis)",
    )
    Chem.MolToMolFile(product, str(OUT / f"B6_q1_{label}_cis.mol"))

(OUT / "rdkit_template_manifest.txt").write_text(
    "Authoritative arms: A1=(V1,V2), A2=(V3,V4), A3=(V5,V6), "
    "A4=(V7,V8), A5=(V9,V10), A6=(V11,V12).\n"
    "q0 fixed: V1-V12; q1 representatives: V2-V11 and V2-V3.\n"
    "V2-V4 is chemically degenerate with V2-V3 and is not duplicated.\n",
    encoding="utf-8",
)
print("wrote", OUT)
