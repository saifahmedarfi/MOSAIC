"""Phase 2 component reimplementation of nine GuacaMol multi-property oracles (spec 5.1).

Each oracle is a scalar geometric mean of component objectives. To score runs on a Pareto
basis we need the components, and we must PROVE we recovered them: `run_gate` compares our
scalar against the benchmark's own oracle on a probe set that is able to fail.

Provenance. Every functional form below was transcribed from BenevolentAI/guacamol (master):
  * modifiers          guacamol/score_modifier.py     (GaussianModifier, MinMaxGaussianModifier,
                                                       ClippedScoreModifier)
  * fingerprints       guacamol/utils/fingerprints.py (ECFP4/6 = count Morgan radius 2/3,
                                                       FCFP4 = same with useFeatures, AP = count
                                                       atom pairs maxLength=10)
  * descriptors        guacamol/utils/descriptors.py  (MolLogP, TPSA, CalcNumRings,
                                                       CalcNumAromaticRings; num_atoms and the
                                                       'H' atom count include explicit Hs)
  * formula parsing    guacamol/utils/chemistry.py    (regex ([A-Z][a-z]*)(\\d*))
  * oracle definitions guacamol/standard_benchmarks.py, common_scoring_functions.py
  * geometric mean     guacamol/utils/math.py         (prod ** (1/n), unweighted)

TRAP. GuacaMol defines two functions that both display the name "Ranolazine MPO":
`ranolazine_mpo` (geometric mean, four components, what is implemented here) and
`start_pop_ranolazine` (arithmetic mean, different components). If TDC bound the latter, a
hand-written reimplementation would disagree silently; the gate turns that into a recorded
exclusion rather than a wrong number.

This module imports rdkit and numpy only. TDC is never imported: `run_gate` takes the TDC
oracle as an argument, so the module loads on machines without it.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

import phase2_stats as _stats


class Component(NamedTuple):
    name: str
    fn: Callable[[Chem.Mol], float]      # mol -> component score in [0, 1]


# --- modifiers (guacamol/score_modifier.py) ---------------------------------------------------
def _gaussian(mu: float, sigma: float) -> Callable[[float], float]:
    return lambda x: float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def _min_gaussian(mu: float, sigma: float) -> Callable[[float], float]:
    """1 for x <= mu, Gaussian decay above."""
    g = _gaussian(mu, sigma)
    return lambda x: g(max(x, mu))


def _max_gaussian(mu: float, sigma: float) -> Callable[[float], float]:
    """1 for x >= mu, Gaussian decay below."""
    g = _gaussian(mu, sigma)
    return lambda x: g(min(x, mu))


def _clipped(upper_x: float, lower_x: float = 0.0) -> Callable[[float], float]:
    """Linear ramp from 0 at lower_x to 1 at upper_x, clipped to [0, 1]."""
    slope = 1.0 / (upper_x - lower_x)
    intercept = 1.0 - slope * upper_x
    return lambda x: float(np.clip(slope * x + intercept, 0.0, 1.0))


# --- descriptors (guacamol/utils/descriptors.py) ----------------------------------------------
def _logp(m): return Descriptors.MolLogP(m)
def _tpsa(m): return Descriptors.TPSA(m)
def _num_rings(m): return rdMolDescriptors.CalcNumRings(m)
def _num_arom(m): return rdMolDescriptors.CalcNumAromaticRings(m)
def _num_atoms(m): return Chem.AddHs(m).GetNumAtoms()          # hydrogens included


def _atom_counter(element: str) -> Callable[[Chem.Mol], int]:
    def f(m):
        if element == "H":
            m = Chem.AddHs(m)
        return sum(1 for a in m.GetAtoms() if a.GetSymbol() == element)
    return f


# --- fingerprints (guacamol/utils/fingerprints.py): legacy COUNT-based vectors -----------------
def _fp(mol: Chem.Mol, kind: str):
    with rdBase.BlockLogs():             # the legacy Morgan API logs a deprecation notice
        if kind == "AP":
            return AllChem.GetAtomPairFingerprint(mol, maxLength=10)
        radius = {"ECFP4": 2, "ECFP6": 3, "FCFP4": 2}[kind]
        return AllChem.GetMorganFingerprint(mol, radius, useFeatures=(kind == "FCFP4"))


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    """guacamol.utils.chemistry.smiles_to_rdkit_mol: parse, then sanitise, None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    return mol


def _tanimoto(ref_smiles: str, kind: str, modifier: Optional[Callable] = None) -> Component:
    ref_fp = _fp(_parse_mol(ref_smiles), kind)
    mod = modifier or (lambda x: x)
    return Component(f"{kind}_sim", lambda m: float(mod(TanimotoSimilarity(_fp(m, kind), ref_fp))))


def _prop(name: str, desc: Callable, modifier: Callable) -> Component:
    return Component(name, lambda m: float(modifier(desc(m))))


def _parse_formula(formula: str) -> List[Tuple[str, int]]:
    """guacamol.utils.chemistry.parse_molecular_formula."""
    return [(el, 1 if not n else int(n)) for el, n in re.findall(r"([A-Z][a-z]*)(\d*)", formula)]


def _isomer(formula: str) -> Component:
    """IsomerScoringFunction: ONE component, itself the geometric mean of a Gaussian(sigma=1)
    per element count and a Gaussian(sigma=2) on the total atom count (hydrogens included)."""
    occ = _parse_formula(formula)
    total = sum(n for _, n in occ)
    parts = [(_atom_counter(el), _gaussian(n, 1.0)) for el, n in occ] + [(_num_atoms, _gaussian(total, 2.0))]
    return Component(f"isomer_{formula}",
                     lambda m: _geomean([mod(desc(m)) for desc, mod in parts]))


def _geomean(values) -> float:
    return float(np.array(values, dtype=float).prod() ** (1.0 / len(values)))


# --- the nine oracles (guacamol/standard_benchmarks.py) ---------------------------------------
_REF = {
    "amlodipine": r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC",
    "fexofenadine": "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
    "osimertinib": "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34",
    "perindopril": "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC",
    "ranolazine": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2",
    "sitagliptin": "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F",
    "zaleplon": "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1",
    "menthol": "CC(C)C1CCC(C)CC1O",
    "camphor": "CC1(C)C2CCC1(C)C(=O)C2",
    "tadalafil": "O=C1N(CC(N2C1CC3=C(C2C4=CC5=C(OCO5)C=C4)NC6=C3C=CC=C6)=O)C",
    "sildenafil": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
}
# Sitagliptin's logP/TPSA targets are computed at runtime from the reference, not hard-coded.
_SITA = Chem.MolFromSmiles(_REF["sitagliptin"])

def _build_components() -> Dict[str, Tuple[Component, ...]]:
    """Built by a function so the mutation tests can rebuild the table under a patched modifier."""
    return {
        "amlodipine_mpo": (
            _tanimoto(_REF["amlodipine"], "ECFP4"),
            _prop("num_rings", _num_rings, _gaussian(3, 0.5))),
        "fexofenadine_mpo": (
            _tanimoto(_REF["fexofenadine"], "AP", _clipped(0.8)),
            _prop("tpsa", _tpsa, _max_gaussian(90, 10)),
            _prop("logp", _logp, _min_gaussian(4, 1))),
        "osimertinib_mpo": (
            _tanimoto(_REF["osimertinib"], "FCFP4", _clipped(0.8)),
            _tanimoto(_REF["osimertinib"], "ECFP6", _min_gaussian(0.85, 0.1)),
            _prop("tpsa", _tpsa, _max_gaussian(100, 10)),
            _prop("logp", _logp, _min_gaussian(1, 1))),
        "perindopril_mpo": (
            _tanimoto(_REF["perindopril"], "ECFP4"),
            _prop("num_aromatic_rings", _num_arom, _gaussian(2, 0.5))),
        # geometric `ranolazine_mpo`, NOT `start_pop_ranolazine` (see module docstring).
        "ranolazine_mpo": (
            _tanimoto(_REF["ranolazine"], "AP", _clipped(0.7)),
            _prop("logp", _logp, _max_gaussian(7, 1)),
            _prop("n_fluorine", _atom_counter("F"), _gaussian(1, 1.0)),
            _prop("tpsa", _tpsa, _max_gaussian(95, 20))),
        "sitagliptin_mpo": (
            _tanimoto(_REF["sitagliptin"], "ECFP4", _gaussian(0, 0.1)),
            _prop("logp", _logp, _gaussian(_logp(_SITA), 0.2)),
            _prop("tpsa", _tpsa, _gaussian(_tpsa(_SITA), 5)),
            _isomer("C16H15F6N5O")),
        "zaleplon_mpo": (
            _tanimoto(_REF["zaleplon"], "ECFP4"),
            _isomer("C19H17N3O2")),
        "median1": (
            _tanimoto(_REF["menthol"], "ECFP4"),
            _tanimoto(_REF["camphor"], "ECFP4")),
        "median2": (
            _tanimoto(_REF["tadalafil"], "ECFP6"),
            _tanimoto(_REF["sildenafil"], "ECFP6")),
    }


COMPONENTS: Dict[str, Tuple[Component, ...]] = _build_components()

# Recorded per oracle, not as one function: if TDC's version of some oracle diverges from
# GuacaMol's, that oracle fails the gate and is excluded with its deviation on record, which
# reads correctly as a spec divergence rather than a bug in our decomposition.
AGGREGATION: Dict[str, str] = {o: "geometric" for o in COMPONENTS}


def component_vector(oracle: str, smiles: str) -> np.ndarray:
    """Each component of `oracle` on `smiles`, in [0, 1]. Raises KeyError for an unknown oracle
    and ValueError for an invalid SMILES (GuacaMol scores those -1; we refuse to guess)."""
    comps = COMPONENTS[oracle]
    mol = _parse_mol(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return np.array([c.fn(mol) for c in comps], dtype=float)


def scalar(oracle: str, smiles: str) -> float:
    """Unweighted geometric mean of the component vector (guacamol.utils.math.geometric_mean)."""
    if AGGREGATION[oracle] != "geometric":
        raise ValueError(f"unsupported aggregation {AGGREGATION[oracle]!r}")
    return _geomean(component_vector(oracle, smiles))


# --- probe set (spec 5.1) ---------------------------------------------------------------------
_REFS: Dict[str, Tuple[str, ...]] = {
    "amlodipine_mpo": (_REF["amlodipine"],), "fexofenadine_mpo": (_REF["fexofenadine"],),
    "osimertinib_mpo": (_REF["osimertinib"],), "perindopril_mpo": (_REF["perindopril"],),
    "ranolazine_mpo": (_REF["ranolazine"],), "sitagliptin_mpo": (_REF["sitagliptin"],),
    "zaleplon_mpo": (_REF["zaleplon"],),
    "median1": (_REF["menthol"], _REF["camphor"]),
    "median2": (_REF["tadalafil"], _REF["sildenafil"]),
}

# Low-end sample. The study's ChEMBL prior is not available to this module (no network, no
# model), so the low end is a frozen literal list of drug-like molecules. Fixed text is more
# reproducible than a sampler; the generating seed below only fixes the order.
_PRIOR_SEED = 20261120
_PRIOR_POOL: Tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1",
    "CN1CCC[C@H]1c1cccnc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "OC(=O)c1ccccc1O",
    "CC(C)NCC(O)COc1cccc2ccccc12", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
    "Clc1ccc2Nc3ccccc3C(=Nc2c1)N1CCNCC1", "CN(C)CCCN1c2ccccc2CCc2ccccc12",
    "O=C(O)c1ccccc1Nc1cccc(c1)C(F)(F)F", "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O", "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "COc1ccc(CCN(C)CCCC(C#N)(C(C)C)c2ccc(OC)c(OC)c2)cc1OC", "OC(CNC1CCCC1)c1ccc(O)c(O)c1",
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "Cc1ccccc1OCC(O)CNC(C)C", "O=C1CCCN1CC(=O)N",
    "NC1=NC(=O)C=CN1C1OC(CO)C(O)C1O", "CCOC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+]([O-])=O)C(=O)OC",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)NC1CCCCC1", "c1ccc2c(c1)ccc1ccccc12", "c1ccc(cc1)-c1ccccc1",
    "c1ccc(cc1)Cc1ccccc1", "c1ccc(cc1)C(=O)c1ccccc1", "Oc1ccc(cc1)-c1ccc(O)cc1",
    "Cc1ccc(cc1)-c1ccc(C)cc1", "c1ccc(cc1)Oc1ccccc1", "c1ccc(cc1)Nc1ccccc1",
    "Clc1ccc(cc1)-c1ccc(Cl)cc1", "c1ccc2[nH]c3ccccc3c2c1", "c1ccc(cc1)-c1ccccn1",
    "c1ccc(cc1)-c1cccnc1", "c1ccc2ncccc2c1", "c1ccc2c(c1)[nH]c1ccccc12",
    "Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1", "O=C(Nc1ccccc1)c1ccccc1", "O=C(Oc1ccccc1)c1ccccc1",
    "c1ccc(cc1)C=Cc1ccccc1", "Nc1ccc(cc1)-c1ccc(N)cc1", "COc1ccc(cc1)-c1ccc(OC)cc1",
    "c1ccc(cc1)C(c1ccccc1)N1CCNCC1", "Cc1ccc(cc1)C(=O)c1ccc(C)cc1",
    "OC(=O)CCCCC(=O)O", "CCCCCCCC", "CCO", "CC(C)O", "CCCCO", "C1CCCCC1", "C1CCNCC1",
    "C1COCCN1", "CC(=O)C", "CCOC(C)=O", "NCCO", "OCC(O)CO", "CC(C)CC(N)C(=O)O",
    "CC1CCC(C(C)C)C(C1)OC(C)=O", "CC1=CCC(CC1)C(C)=C", "CC(C)=CCCC(C)=CC=O",
    "O=C1CCC(CC1)C(C)(C)C", "CC12CCC3C(CCc4cc(O)ccc34)C1CCC2O", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
    "CC(C)C1CCC(C)CC1=O", "CC1(C)C2CCC1(C)C(O)C2", "CC1(C)C2CCC1(C)C(=O)C2C",
    "O=C1N(C)C(=O)c2ccccc12", "CC1CCC2(CC1)OCCO2", "OC1CCC(CC1)c1ccccc1",
    "FC(F)(F)c1ccccc1", "Fc1ccc(cc1)C(F)(F)F", "Fc1cc(F)cc(F)c1", "FC(F)(F)CN1CCCC1",
    "N#Cc1ccc2ncccc2c1", "CCN(C(C)=O)c1ccccc1", "O=C(C)N(CC)c1cccc(c1)-c1ccncc1",
    "OC(=O)C(C)(C)c1ccc(cc1)C(O)CCCN1CCCCC1", "CC(C)(C(=O)O)c1ccc(cc1)C(=O)C",
    "NC(CC(=O)N1CCN2C(C1)=NN=C2)Cc1cc(F)c(F)cc1F", "NC(CC(=O)N1CCNCC1)Cc1ccccc1",
    "CCCC1=NN(C)C2=C1N=C(NC2=O)c1ccccc1", "COc1ccccc1OCC(O)CNCCc1ccccc1",
    "COc1ccccc1OCC(O)CN1CCN(CC1)c1ccccc1", "O=C(Cc1ccccc1)Nc1c(C)cccc1C",
    "CN(C)CCN(C)c1ccccc1NC(=O)C=C", "c1cn(C)c2ccccc12", "c1ccc(cc1)Nc1nccc(n1)-c1ccccc1",
    "COc1cc(N(C)C)c(N)cc1Nc1nccc(n1)-c1cn(C)c2ccccc12", "O=C1COC(=O)N1",
    "O=C1NC(=O)C(=O)N1", "Cc1cccc(C)c1NC(=O)CN1CCN(C)CC1", "CCOC(=O)C(CCc1ccccc1)NC(C)C(=O)N1CCCC1C(=O)O",
    "CCCC(NC(C)C(=O)N1C2CCCCC2CC1C(=O)O)C(=O)OCC", "CCCC(NC(C)C(=O)N1CCCC1C(=O)O)C(=O)OCC",
    "ClC1=CC=CC=C1C1C(C(=O)OC)=C(C)NC(C)=C1C(=O)OC", "O=C1c2ccccc2C(=O)N1CCOCCN",
    "c1ccc2c(c1)OCO2", "O=C1CC2=C(N1)C=CC=C2", "C1=CC2=C(C=C1)C=CC=C2", "C1=CC=C2C(=C1)C=CN2",
    # lipophilic extreme: a half-Gaussian logP target at 7 (ranolazine) is unreachable otherwise
    "CCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC", "CCCCCCCCCCCCCCCCCCCC(=O)OC",
    "CCCCCCCCCCCCc1ccc(cc1)-c1ccccc1", "Clc1cc(Cl)c(c(Cl)c1)-c1c(Cl)cc(Cl)cc1Cl",
    "CC(C)CCCC(C)CCCC(C)CCCC(C)C", "c1ccc(cc1)C(c1ccccc1)(c1ccccc1)c1ccccc1",
    "CCCCCCCCCCCCCCCCN1CCCCC1",
    "CC(C)c1ccc(cc1)C(C)C", "CCOc1ccc(cc1)C(=O)OCC", "OCc1ccccc1", "O=Cc1ccc(O)c(OC)c1",
    "Nc1ccc(cc1)S(N)(=O)=O", "CC(=O)Nc1ccc(cc1)S(N)(=O)=O", "CS(=O)(=O)c1ccc(cc1)-c1ccccc1",
)


def _neighbours(ref_smiles: str) -> List[str]:
    """Systematic single-edit neighbours of a reference: heteroatom swaps, leaf-atom deletions,
    methyl/F/Cl/O/N additions at every atom with a free valence, and ring-bond cleavages.
    Deterministic (atom order is the SMILES order); invalid edits are dropped."""
    base = Chem.MolFromSmiles(ref_smiles)
    out: List[str] = []

    def emit(rw: Chem.RWMol):
        try:
            with rdBase.BlockLogs():         # invalid edits are expected and simply dropped
                m = rw.GetMol()
                Chem.SanitizeMol(m)
                s = Chem.MolToSmiles(m)
                ok = Chem.MolFromSmiles(s) is not None
            if ok:
                out.append(s)
        except (ValueError, RuntimeError):   # RDKit raises these for invalid edits; anything else is a bug
            pass

    for a in base.GetAtoms():
        i = a.GetIdx()
        for z in (6, 7, 8, 9, 16, 17):                        # swap the element
            if z != a.GetAtomicNum() and not (a.GetIsAromatic() and z in (9, 17)):
                rw = Chem.RWMol(base); rw.GetAtomWithIdx(i).SetAtomicNum(z)
                rw.GetAtomWithIdx(i).SetNoImplicit(False); emit(rw)
        if a.GetDegree() == 1:                                 # delete a leaf atom
            rw = Chem.RWMol(base); rw.RemoveAtom(i); emit(rw)
        if not a.GetIsAromatic():                              # grow a substituent
            for z in (6, 7, 8, 9, 17):
                rw = Chem.RWMol(base); j = rw.AddAtom(Chem.Atom(z)); rw.AddBond(i, j, Chem.BondType.SINGLE); emit(rw)
    for b in base.GetBonds():
        if b.IsInRing() and not b.GetIsAromatic():             # open a ring
            rw = Chem.RWMol(base); rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx()); emit(rw)
    return out


def probe_set(oracle: str) -> List[str]:
    """Frozen, deterministic probe molecules: the oracle's own reference(s), single-edit
    neighbours of them (so scores are NOT all zero), and a fixed prior-like sample for the low
    end. Canonical SMILES, de-duplicated, order fixed."""
    refs = _REFS[oracle]
    seen, out = set(), []

    def add(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return
        c = Chem.MolToSmiles(m)
        if c not in seen:
            seen.add(c); out.append(c)

    for r in refs:
        add(r)
    for r in refs:
        for s in _neighbours(r):
            add(s)
    pool = list(_PRIOR_POOL)
    order = np.random.default_rng(_PRIOR_SEED).permutation(len(pool))
    for k in order:
        add(pool[k])
    return out


def run_gate(oracle: str, tdc_oracle) -> Dict[str, Any]:
    """Score the probe set through our components and through `tdc_oracle` (a callable taking a
    list of SMILES, e.g. tdc.Oracle(name=...)), and return `gate_decision` plus bookkeeping."""
    smiles = probe_set(oracle)
    comps = np.array([component_vector(oracle, s) for s in smiles])
    ours = np.array([scalar(oracle, s) for s in smiles])
    theirs = np.asarray(tdc_oracle(smiles), dtype=float)
    g = _stats.gate_decision(ours, theirs, comps)
    g["oracle"] = oracle
    g["n_probe"] = len(smiles)
    g["aggregation"] = AGGREGATION[oracle]
    return g
