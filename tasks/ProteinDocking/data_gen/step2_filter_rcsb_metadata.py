"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import os
import time
import gzip
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from urllib.request import urlopen
from functools import partialmethod
from collections import defaultdict
from Bio.PDB.PDBParser import PDBParser

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def filter_rcsb_metadata():

    # entry metadata
    entry_df = pd.read_csv('entry_metadata.csv.gz', header=0, index_col=None, sep=',')
    entry_df = entry_df.loc[entry_df['resolution_combined'] < 3.5]

    # assembly metadata
    assembly_df = pd.read_csv('assembly_metadata.csv.gz', header=0, index_col=None, sep=',')
    assembly_df = assembly_df.loc[(assembly_df['assembly_id'] == 1)
                                & (assembly_df['polymer_entity_instance_count'] > 1)
                                & (assembly_df['polymer_entity_instance_count'] < 5)
                                & (assembly_df['entry_id'].isin(entry_df['entry_id']))
                                & (assembly_df['selected_polymer_entity_types'] == 'Protein (only)')]

    # polymer interface metadata
    interface_df = pd.read_csv('polymer_interface_metadata.csv.gz', header=0, index_col=None, sep=',')
    interface_df = interface_df.loc[interface_df['entry_id'].isin(assembly_df['entry_id'])]
    # relabel chain IDs
    entity_instance_df = pd.read_csv('polymer_entity_instance.csv.gz', header=0, index_col=None, sep=',')
    interface_df_tmp = pd.merge(
        left=interface_df, left_on=['entry_id', 'chain_1'],
        right=entity_instance_df[['entry_id', 'asym_id', 'auth_asym_id']], right_on=['entry_id', 'asym_id']
    ).rename(columns={'auth_asym_id': 'auth_chain_1'}).drop(columns=['asym_id'])
    interface_df = pd.merge(
        left=interface_df_tmp, left_on=['entry_id', 'chain_2'],
        right=entity_instance_df[['entry_id', 'asym_id', 'auth_asym_id']],  right_on=['entry_id', 'asym_id']
    ).rename(columns={'auth_asym_id': 'auth_chain_2'}).drop(columns=['asym_id'])
    hetero_interface_df = interface_df.loc[interface_df['interface_character'] == 'hetero']

    return hetero_interface_df


def filter_pdb_data(pdb_data_dir, entry_id):
    fname = entry_id.lower() + '.pdb1.gz'
    fpath = os.path.join(pdb_data_dir, entry_id[1:3].lower(), fname)
    if not os.path.isfile(fpath):
        print('missing file', fpath)
        return None

    parser = PDBParser(PERMISSIVE=1) 
    with gzip.open(fpath, 'rt') as handle:
        struct = parser.get_structure(id=entry_id, file=handle)
    
    # filter structure method
    """
    'x-ray diffraction', 'solution nmr; theoretical model', 'neutron diffraction; x-ray diffraction', 
    'electron crystallography', 'solution scattering', 'solution nmr; solution scattering', 'solution nmr', 
    'electron microscopy', 'x-ray diffraction; neutron diffraction', 'solid-state nmr', 'neutron diffraction'
    """
    struct_method = struct.header['structure_method']
    if not ('diffraction' in struct_method or struct_method == 'electron microscopy'):
        return None
    
    # skip multi-model complexes
    model_list = list(struct.get_models())
    if len(model_list) > 1:
        return None

    return entry_id


def filter_by_entity(rcsb_ids, db5_ids, threshold):
    if not os.path.isfile(f'dict1_{threshold}.pkl'):
        # fetch entity clustering info
        url = f'https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{threshold}.txt'
        dict1 = defaultdict(list)
        dict2 = dict()
        input_pdb_ids = set(rcsb_ids + db5_ids)
        cnt = 0
        for _, line in tqdm(enumerate(urlopen(url))):
            decoded_line = line.decode('utf-8')
            pdb_ids = [s.split('_')[0] for s in decoded_line.strip('\n').split(' ')
                    if s.split('_')[0] in input_pdb_ids]
            if len(pdb_ids) < 1:
                continue
            pdb_ids = list(set(pdb_ids))
            for pdb_id in pdb_ids:
                dict1[pdb_id].append(cnt)
            dict2[cnt] = pdb_ids
            cnt += 1
        # save dictionary
        with open(f'dict1_{threshold}.pkl', 'wb') as f1:
            pickle.dump(dict1, f1)
        with open(f'dict2_{threshold}.pkl', 'wb') as f2:
            pickle.dump(dict2, f2)
    else:
        with open(f'dict1_{threshold}.pkl', 'rb') as f1:
            dict1 = pickle.load(f1)
        with open(f'dict2_{threshold}.pkl', 'rb') as f2:
            dict2 = pickle.load(f2)
    
    # create group mapping matrix
    num_pdb_ids = len(dict1.keys())
    assert num_pdb_ids == len(list(set(dict1.keys())))
    num_groups = len(dict2.keys())
    all_pdb_ids = list(dict1.keys())
    group_map = np.zeros((num_pdb_ids, num_groups))
    for key, val in tqdm(dict1.items()):
        irow = all_pdb_ids.index(key)
        icol = np.array(val, dtype=int)
        group_map[irow, icol] = 1
    group_corr = group_map @ group_map.T

    similar_ids = []
    for pid in tqdm(db5_ids):
        if pid not in all_pdb_ids: # e.g., 3RVW was recently removed from RCSB
            continue
        idx = all_pdb_ids.index(pid)
        similar_ids.extend(list(np.where(group_corr[idx] > 0)[0]))
    similar_ids = set(similar_ids)
    print('size of similar entries to DB5:', len(similar_ids))
    
    filtered_ids = [pid for pid in all_pdb_ids if all_pdb_ids.index(pid) not in similar_ids]
    
    return filtered_ids


def extract_pairs(hetero_interface_df):
    valid_pairs = []
    for entry_id in tqdm(hetero_interface_df['entry_id'].unique()):
        interfaces = hetero_interface_df.loc[hetero_interface_df['entry_id'] == entry_id]
        iface_chain_ids = []
        for _, iface in interfaces.iterrows():
            iface_chain_ids.append(iface['auth_chain_1'])
            iface_chain_ids.append(iface['auth_chain_2'])
        iface_chain_ids = list(set(iface_chain_ids))
        if len(iface_chain_ids) == 2:
            chains = '_'.join(sorted(iface_chain_ids))
            valid_pairs.append(f'{entry_id}_{chains}')
        else:
            for lig_chain_id in iface_chain_ids:
                rec_chain_ids = ''.join(sorted([i for i in iface_chain_ids if i != lig_chain_id]))
                valid_pairs.append(f'{entry_id}_{lig_chain_id}_{rec_chain_ids}')
        
    return valid_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--mute-tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    # optionally mute tqdm
    if args.mute_tqdm:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    start = time.time()

    # obtain original PDB IDs
    entry_df = pd.read_csv('entry_metadata.csv.gz', header=0, index_col=None, sep=',')
    rcsb_ids = entry_df['entry_id'].unique().tolist()

    # filter metadata for heteromeric PDB IDs
    hetero_interface_df = filter_rcsb_metadata()
    hetero_pdb_ids = set(hetero_interface_df['entry_id'])
    rcsb_ids = [pid for pid in rcsb_ids if pid in hetero_pdb_ids]
    
    # remove RCSB IDs that are highly similar to DB5, while DIPS has already been filtered
    db5_df = pd.read_csv('db5_difficulty.csv', header=0, index_col=None, sep=',')
    db5_pdb_ids = list(db5_df['pdb_id'].unique())
    assert len(db5_pdb_ids) == 253
    rcsb_ids = filter_by_entity(rcsb_ids=rcsb_ids, db5_ids=db5_pdb_ids, threshold=30)
    
    # filter PDB data
    pdb_data_dir = './RCSB_pdb/'
    if not args.serial:
        pool = multiprocessing.Pool(processes=args.j)
        pool_args = [(pdb_data_dir, entry_id) for entry_id in rcsb_ids]
        ret = pool.starmap(filter_pdb_data, tqdm(pool_args), chunksize=10)
        pool.terminate()
        valid_ids = [fpath for fpath in ret if fpath is not None]
    else:
        valid_ids = []
        for entry_id in tqdm(rcsb_ids):
            ret = filter_pdb_data(pdb_data_dir, entry_id)
            if ret is not None:
                valid_ids.append(ret)

    hetero_interface_df = hetero_interface_df.loc[hetero_interface_df['entry_id'].isin(valid_ids)]
    filtered_pdb_ids = list(hetero_interface_df['entry_id'].unique())
    print(f'number of filtered PDB IDs: {len(filtered_pdb_ids)}')
    
    # extract pairs
    valid_pairs = extract_pairs(hetero_interface_df)
    print('number of valid pairs:', len(valid_pairs))

    out_fpath = 'valid_pairs.txt'
    with open(out_fpath, 'w') as f:
        for ipair in valid_pairs:
            f.write(ipair + '\n')

    print(f'RCSB step2 elapsed time: {(time.time()-start):.1f}s\n')


