"""Chemical properties"""

# atomic van der waals radii in Angstrom unit (from mendeleev)
vdw_radii_dict = { 
    'H':   1.1,
    'C':   1.7,
    'N':   1.55,
    'O':   1.52,
    'S':   1.8,
    'F':   1.47,
    'P':   1.8,
    'Cl':  1.75,
    'Se':  1.9,
    'Br':  1.85,
    'I':   1.98,
    'UNK': 2.0,
}

# atom type label for one-hot-encoding
atom_type_dict = {
    'H':  0,
    'C':  1,
    'N':  2,
    'O':  3,
    'S':  4,
    'F':  5,
    'P':  6,
    'Cl': 7,
    'Se': 8,
    'Br': 9,
    'I':  10,
    'UNK': 11,
}

# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0,
    'GLY': 1,
    'SER': 2,
    'THR': 3,
    'LEU': 4,
    'ILE': 5,
    'VAL': 6,
    'ASN': 7,
    'GLN': 8,
    'ARG': 9,
    'HIS': 10,
    'TRP': 11,
    'PHE': 12,
    'TYR': 13,
    'GLU': 14,
    'ASP': 15,
    'LYS': 16,
    'PRO': 17,
    'CYS': 18,
    'MET': 19,
    'UNK': 20,
}

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5,
    'VAL': 4.2,
    'LEU': 3.8,
    'PHE': 2.8,
    'CYS': 2.5,
    'MET': 1.9,
    'ALA': 1.8,
    'GLY': -0.4,
    'THR': -0.7,
    'SER': -0.8,
    'TRP': -0.9,
    'TYR': -1.3,
    'PRO': -1.6,
    'HIS': -3.2,
    'GLU': -3.5,
    'GLN': -3.5,
    'ASP': -3.5,
    'ASN': -3.5,
    'LYS': -3.9,
    'ARG': -4.5,
    'UNK': 0.0,
}


res_type_to_hphob = {
    idx: hydrophob_dict[res_type] for res_type, idx in res_type_dict.items()
}
