"""Data IO functions

-----
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import os
import numpy as np
from pathlib import Path
from HMR.data_gen.chemistry import atom_type_dict, res_type_dict
from HMR.utils.helpers import subprocess_run


def parse_pqr_file(pqr_fpath):
    """Read and parse .pqr files

    Args:
        pqr_fpath (path-like): .pqr file

    Returns:
        xyz_list (np.ndarray): coordinates of atoms
        rn_list (list of str): radius and atom id info, formatted as "{radius} 1 {atom_full_id}"
    """

    with open(pqr_fpath, 'r') as f:
        f_read = f.readlines()
    xyz_list = [] # atomic coordinates
    rn_list = [] # radius and descriptions
    for line in f_read:
        if line[:4] == 'ATOM':
            assert (len(line) == 70) and (line[69] == '\n')
            atom_id = int(line[6:11]) # 1-based indexing
            assert line[11] == ' '
            atom_name = line[12:16].strip()
            assert atom_name[0] in atom_type_dict
            assert line[16] == ' '
            res_name = line[17:20]
            if not res_name in res_type_dict:
                res_name = 'UNK'
            res_id = int(line[22:26].strip()) # 1-based indexing
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            assert line[54] == ' '
            charge = float(line[55:62])
            assert line[62] == ' '
            radius = float(line[63:69])
            xyz_list.append([x, y, z])
            full_id = f'{res_name}_{res_id:d}_{atom_name}_{atom_id:d}_{charge:.4f}_{radius:.4f}'
            rn_list.append(str(radius) + ' 1 ' + full_id)

    return np.array(xyz_list, dtype=float), rn_list


def convert_pdb_to_pqr(pdb_fpath, pqr_fpath, pdb2pqr_bin, pdb2pqr_log=None):
    """Convert .pdb file to .pqr file using pdb2pqr

    pdb2pqr: we set --ff=AMBER and other parameters as default
             
    Args:
        pdb_fpath (path): input .pdb file 
        pqr_fpath (path): output .pdb file
        pdb2pqr_bin (path): pdb2pqr binary file
        pdb2pqr_log (path): save pdb2pqr output to file

    Returns:
        0 if success, 1 if error encountered
    """
    
    # IO
    pdb_fpath = Path(pdb_fpath)
    assert pdb_fpath.exists(), f"{str(pdb_fpath)} not found"
    pqr_fpath = Path(pqr_fpath)
    out_dir = pqr_fpath.parent
    assert out_dir.exists(), f"Output directory: {str(out_dir)} doesn't exist"
    pdb_id = pdb_fpath.stem
    assert Path(pdb2pqr_bin).exists(), f"pdb2pqr_bin {str(pdb2pqr_bin)} doesn't exist"
    
    # pdb2pqr
    _, err = subprocess_run(
        [pdb2pqr_bin, '--ff=AMBER', str(pdb_fpath), str(pqr_fpath)],
        print_out=False, out_log=pdb2pqr_log, err_ignore=True
    )
    if 'CRITICAL' in err:
        print(f'{pdb_id} pdb2pqr failed', flush=True)
        if pdb2pqr_log is not None:
            with open(pdb2pqr_log, 'a') as handle:
                handle.write('\n' + 'pdb2pqr Error:')
                handle.write(err)
        assert not pqr_fpath.exists()
        return 1
    return 0


def convert_pqr_to_xyzrn(pqr_fpath, xyzrn_fpath):
    """Convert .pqr file to .xyzrn file

    Args:
        pqr_fpath (path): input .pqr file
        xyzrn_fpath (path): output .xyzrn file
    """
    # IO
    assert Path(pqr_fpath).exists()
    out_dir = Path(xyzrn_fpath).parent
    assert out_dir.exists(), f"Output directory: {str(out_dir)} doesn't exist"

    # pqr to xyzrn    
    xyz, rn = parse_pqr_file(pqr_fpath)
    with open(xyzrn_fpath, 'w') as f:
        for idx in range(len(xyz)):
            coords = '{:.6f} {:.6f} {:.6f} '.format(*xyz[idx])
            f.write(coords + rn[idx] + '\n')


def read_xyzrn_file(fpath):
    """Read xyzrn file and extract atomic features
    
    Returns:
        np.ndarray: (N, 8) array, each row include the atom info of
            [x, y, z, res_type, atom_type, charge, radius, is_alpha_carbon]
    """

    assert os.path.isfile(fpath), f"{fpath} not found"
    atom_info = []
    with open(fpath, 'r') as f:
        for line in f.readlines(): 
            line_info = line.rstrip().split()
            assert len(line_info) == 6
            full_id = line_info[-1]
            assert len(full_id.split('_')) == 6
            res_name, res_id, atom_name, atom_id, charge, radius = full_id.split('_')

            assert res_name in res_type_dict
            assert atom_name[0] in atom_type_dict
            alpha_carbon = atom_name.upper() == 'CA'
            atom_info.append(
                line_info[:3] + 
                [res_type_dict[res_name],
                 atom_type_dict[atom_name[0]],
                 float(charge),
                 float(radius),
                 alpha_carbon]
            )

    return np.array(atom_info, dtype=float)
