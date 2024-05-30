import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem import rdqueries  # pylint: disable=unused-import


class ADME:
    """
    ADME is the main class that contains the core ADME calculations. Functions
    include many druglikeness filters (Lipinski's rule of five, Egan, Ghose,
    BOILED-Egg, etc.), medicinal chemistry filters (PAINS, Brenk), and provides
    a consistent, simple wrapping API for common cheminformatics calculations.
    """

    # Class constants

    def __init__(self, _mol):
        if isinstance(_mol, str):
            self.mol = Chem.MolFromSmiles(_mol)
        else:
            self.mol = _mol

    def druglikeness_egan(self, verbose=False):
        """
        Egan (2000) Prediction of Drug Absorption Using Multivariate Statistics

        The Egan paper creates the "Egan egg" a multivariable ellipse model.
        For simplicity, we don't implement the actual Egan egg here, but rather
        the simpler rule based approximation that the paper describes as the
        upper bounds for their model inputs. We use the 95% confidence interval
        levels here, rather than the 99% confidence intervals.

        In the future the full Egan egg model should be implemented here, but
        that is low priority given the problems with the Egan model.
        """
        violations = []

        tpsa = self._tpsa()
        if tpsa > 131.6:
            violations.append(f"PSA {tpsa}")

        logp = self._logp()
        if logp > 5.88:
            violations.append(f"logP {logp}")

        if verbose:
            return violations

        return len(violations) < 1

    def druglikeness_ghose(self, verbose=False):
        """
        Ghose (1999) A Knowledge-Based Approach in Designing Combinatorial or
        Medicinal Chemistry Libraries for Drug Discovery.
        1. A Qualitative and Quantitative Characterization of Known Drug
        Databases

        In this paper (and this function) they define the qualifying range
        (the chance of missing good compounds is less than 20%) as a certain
        range of Log P, molecular weight, molar refractivity, and the total
        number of atoms. They also define a preferred range as the  interval
        having 50% of drugs. The tighter preferred range is implemented in
        druglikeness_ghose_pref.
        """
        violations = []

        logp = self._logp()
        if logp > 5.6 or logp < -0.4:
            violations.append(f"LOGP {logp}")

        molecular_weight = self._molecular_weight()
        if molecular_weight < 160 or molecular_weight > 480:
            violations.append(f"Molecular Mass {molecular_weight}")

        molar_refractivity = self._molar_refractivity()
        if molar_refractivity < 40 or molar_refractivity > 130:
            violations.append(f"MR {molar_refractivity}")

        n_atoms = self._n_atoms()
        if n_atoms < 20 or n_atoms > 70:
            violations.append(f"N Atoms {n_atoms}")

        if verbose:
            return violations

        return len(violations) < 1

    def druglikeness_ghose_pref(self, verbose=False):
        """
        Ghose (1999) A Knowledge-Based Approach in Designing Combinatorial or
        Medicinal Chemistry Libraries for Drug Discovery.
        1. A Qualitative and Quantitative Characterization of Known Drug
        Databases

        In this paper they define the qualifying range (the chance of missing
        good compounds is less than 20%) as a certain range of Log P, molecular
        weight, molar refractivity, and the total number of atoms. They define
        a preferred range (implemented in this function) as the  interval
        having 50% of drugs. The looser qualifying range is implemented in
        druglikeness_ghose.
        """
        violations = []

        logp = self._logp()
        if logp > 4.1 or logp < 1.3:
            violations.append(f"LOGP {logp}")

        molecular_weight = self._molecular_weight()
        if molecular_weight < 230 or molecular_weight > 390:
            violations.append(f"Molecular Mass {molecular_weight}")

        molar_refractivity = self._molar_refractivity()
        if molar_refractivity < 70 or molar_refractivity > 110:
            violations.append(f"MR {molar_refractivity}")

        n_atoms = self._n_atoms()
        if n_atoms < 30 or n_atoms > 55:
            violations.append(f"N Atoms {n_atoms}")

        if verbose:
            return violations

        return len(violations) < 1

    def druglikeness_lipinski(self, verbose=False):
        """
        Lipinski (2001) Experimental and computational approaches to estimate
        solubility and permeability in drug discovery and development settings

        https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five

        Lipinski's rule of 5 is one of the most important druglikenss filters,
        against which all others are judged. The rules of the filter are no
        more than 5 hydrogen bond donors, no more than 10 hydrogen bond
        acceptors, a molecular mass of less than 500 daltons, and a logP
        that does not exceed 5.
        """
        violations = []

        h_bond_donors = self._h_bond_donors()
        if h_bond_donors > 5:
            violations.append(f"H Bond Donors {h_bond_donors}>5")

        h_bond_acceptors = self._h_bond_acceptors()
        if h_bond_acceptors > 10:
            violations.append(f"H Bond Acceptors {h_bond_acceptors}>10")

        molecular_weight = self._molecular_weight()
        if molecular_weight > 500:
            violations.append(f"Molecular Weight {molecular_weight}>500")

        logp = self._logp()
        if logp > 5:
            violations.append(f"LOGP {logp}>5")

        if verbose:
            if not violations:
                return "No violations found"
            return violations

        return len(violations) < 1

    def druglikeness_muegge(self, verbose=False):
        """
        Muegge (2001) Simple Selection Criteria for Drug-like Chemical Matter

        In this paper they define a few specific "pharmacophore points" that
        are essentially proxies of hydrogen bonding ability. SwissADME uses
        a few simple rules based filters to approximate the pharmacophore point
        filter in this paper, and we implement the rules based filter here.
        """
        violations = []

        molecular_weight = self._molecular_weight()
        if molecular_weight > 600 or molecular_weight < 200:
            violations.append(f"MW {molecular_weight}")

        logp = self._logp()
        if logp > 5 or logp < -2:
            violations.append(f"LOGP {logp}")

        tpsa = self._tpsa()
        if tpsa > 150:
            violations.append(f"TPSA {tpsa}")

        n_rings = self._n_rings()
        if n_rings > 7:
            violations.append(f"N Rings {n_rings}")

        n_carbon = self._n_carbons()
        if n_carbon < 5:
            violations.append(f"N Carbon {n_carbon}")

        n_heteroatoms = self._n_heteroatoms()
        if n_heteroatoms < 2:
            violations.append(f"N Heteroatoms {n_heteroatoms}")

        n_rot_bonds = self._n_rot_bonds()
        if n_rot_bonds > 15:
            violations.append(f"N Rot Bonds {n_rot_bonds}")

        h_bond_acc = self._h_bond_acceptors()
        if h_bond_acc > 10:
            violations.append(f"H Bond Acc {h_bond_acc}")

        h_bond_don = self._h_bond_donors()
        if h_bond_don > 5:
            violations.append(f"H Bond Don {h_bond_don}")

        if verbose:
            return violations

        return len(violations) < 1

    def druglikeness_veber(self, verbose=False):
        """
        Veber (2002) Molecular Properties That Influence the Oral
        Bioavailability of Drug Candidates

        This study on oral bioavailability in rats shows that molecular weight
        is only a good predictor of bioavailability insofar as it's correlated
        with rotatable bonds and polar surface area. They actually propose two
        different druglikeness rules that achieve a similar effect:
        i. polar surface area <= 140 angstroms squared AND # rot. bonds <= 10
        OR
        ii. sum of H-bond donors and acceptors <= 12 AND # rot. bonds <= 10

        SwissADME uses the polar surface area metric
        """
        violations = []

        tpsa = self._tpsa()
        if tpsa > 140:
            violations.append(f"TPSA {tpsa}")

        n_rot_bonds = self._n_rot_bonds()
        if n_rot_bonds > 10:
            violations.append(f"N Rotatable Bonds {n_rot_bonds}")

        if verbose:
            return violations

        return len(violations) < 1

    def brenk(self):
        """
        Brenk (2008) Lessons Learnt from Assembling Screening Libraries for
        Drug Discovery for Neglected Diseases

        Brenk's Structural Alert filter finds fragments "putatively toxic,
        chemically reactive, metabolically unstable or to bear properties
        responsible for poor pharmacokinetics."

        Returns:
            Boolean of whether the molecule triggers the Brenk filter.
        """
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog.FilterCatalog(params)
        return catalog.HasMatch(self.mol)

    def pains(self):
        """
        Baell and Holloway (2010) New Substructure Filters for Removal of Pan
        Assay Interference Compounds (PAINS) from Screening Libraries and for
        Their Exclusion in Bioassays

        This filter finds promiscuous compounds that are likely to show activity
        regardless of the target.

        Returns:
            Boolean of whether the molecule triggers the PAINS filter.
        """
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        return catalog.HasMatch(self.mol)

    def _h_bond_donors(self) -> int:
        """Number of Hydrogen Bond Donors"""
        return Chem.Lipinski.NumHDonors(self.mol)

    def _h_bond_acceptors(self) -> int:
        """Number of Hydrogen Bond Acceptors"""
        return Chem.Lipinski.NumHAcceptors(self.mol)

    def _molar_refractivity(self):
        """Wildman-Crippen Molar Refractivity"""
        return Chem.Crippen.MolMR(self.mol)

    def _molecular_weight(self):
        """Molecular weight"""
        return Descriptors.ExactMolWt(self.mol)

    def _n_atoms(self):
        """Number of atoms"""
        return self.mol.GetNumAtoms()

    def _n_carbons(self):
        """Number of carbon atoms"""
        carbon = Chem.rdqueries.AtomNumEqualsQueryAtom(6)
        return len(self.mol.GetAtomsMatchingQuery(carbon))

    def _n_heteroatoms(self):
        """Number of heteroatoms"""
        return Descriptors.rdMolDescriptors.CalcNumHeteroatoms(self.mol)

    def _n_rings(self):
        """Number of rings"""
        return Descriptors.rdMolDescriptors.CalcNumRings(self.mol)

    def _n_rot_bonds(self):
        """Number of rotatable bonds"""
        return Chem.Lipinski.NumRotatableBonds(self.mol)

    def _logp(self):
        """Log of partition coefficient"""
        return Descriptors.MolLogP(self.mol)

    def _tpsa(self):
        """Topological polar surface area"""
        return Descriptors.TPSA(self.mol)

def generate_molecular_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Generate MACCS keys
            mol = ADME(smiles)
            # Calculate molecular descriptors
            tpsa = mol._tpsa()
            logp = mol._logp()
            molecular_weight = mol._molecular_weight()
            molar_refractivity = mol._molar_refractivity()
            n_atoms = mol._n_atoms()
            n_rings = mol._n_rings()
            n_carbon = mol._n_carbons()
            n_heteroatoms = mol._n_heteroatoms()
            n_rot_bonds = mol._n_rot_bonds()
            h_bond_acc = mol._h_bond_acceptors()
            h_bond_don = mol._h_bond_donors()

            return np.array([tpsa, logp, molecular_weight, molar_refractivity, n_atoms,
                                               n_rings, n_carbon, n_heteroatoms, n_rot_bonds,
                                               h_bond_acc, h_bond_don], dtype=float)
        else:
            return None
    except Exception as e:
        print(f"An unexpected error occurred in generate_maccs_keys_molecular_descriptors function: {e}")
        return None

def preprocessing_md(target, assay_type, data):
    try:
        print("Starting molecules fingerprinting: ")
        maccs_keys_descriptors_df = data["canonical_smiles"].apply(generate_molecular_descriptors).dropna().tolist()

        return np.array(maccs_keys_descriptors_df)

    except Exception as e:
        print(f"An unexpected error occurred in preprocessing_mk function: {e}")
        return None
