"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""


import rcsb_api
import pandas as pd
from pathlib import Path
from datetime import datetime
from helpers import subprocess_run, log_info
from argparse import ArgumentParser, RawDescriptionHelpFormatter

def rsync_rcsb(rcsb_root_path, dry_run=True):
    """Rsync RCSB PDB dataset. Changes are write to rcsb_root_path/change-[date].log"""

    rcsb_root_path = Path(rcsb_root_path).resolve()
    print('rsync RCSB PDB dataset...')
    cmd = f"rsync -rlpt -v -z --delete --port=33444 --outbuf=L rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ {str(rcsb_root_path)}/divided"
    if dry_run:
        cmd += ' --dry-run'
    out, err = subprocess_run(cmd)
    if not dry_run:
        log_info(out, logger=rcsb_root_path/f"change-{datetime.now().strftime('%Y%m%d')}.log" )
    

def query_metadata(database_path, metadata_path, j=None):
    """Query and update metadata files, including
        - entry_metadata.csv.gz
        - assembly_metadata.csv.gz
        - polymer_entity_metadata.csv.gz
        - polymer_entity_instance.csv.gz
        - polymer_interface_metadata.csv.gz
        - polymer_entity_instance_annotation.csv.gz
    """

    metadata_path = Path(metadata_path)
    database_path = Path(database_path)
    assert database_path.exists(), f"{str(database_path)} not exists"

    if not metadata_path.exists():
        metadata_path.mkdir(parents=True)
        
    num_pdb_entries = len([
        pdb for block in rcsb_api.get_entry_id_blocks(folder_path=database_path) 
        for pdb in block
    ])

    print(f"{num_pdb_entries} PDB entries found")

    # entry-level metadata
    print('Querying entry-level metadata...')
    entry_metadata_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_entry_metadata, 
        folder_path=database_path,
        id_type='entry',
        j=j
    )
    entry_metadata = pd.concat(entry_metadata_list, axis=0).astype(rcsb_api.entry_metadata_col_dtype)
    print(f"\tEntry coverage: {entry_metadata['entry_id'].nunique() / num_pdb_entries:.2%}")
    entry_metadata.to_csv(metadata_path/'entry_metadata.csv.gz', index=False)

    # assembly-level metadata
    print('Querying assembly-level metadata...')
    assembly_metadata_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_assembly_metadata, 
        folder_path=database_path,
        id_type='entry',
        j=j
    )
    assembly_metadata = pd.concat(assembly_metadata_list, axis=0).astype(rcsb_api.assembly_metadata_col_dtype)
    print(f"\tEntry coverage: {assembly_metadata['entry_id'].nunique() / num_pdb_entries:.2%}")
    assembly_metadata.to_csv(metadata_path/'assembly_metadata.csv.gz', index=False)

    # Polymer entity-level metadata
    print("Querying polymer entity metadata...")
    polymer_entity_metadata_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_polymer_entity_metadata, 
        folder_path=database_path,
        id_type='entry',
        j=j
    )
    polymer_entity_metadata = pd.concat(polymer_entity_metadata_list, axis=0).astype(rcsb_api.polymer_entity_col_dtype)
    print(f"\tEntry coverage: {polymer_entity_metadata['entry_id'].nunique() / num_pdb_entries:.2%}")
    polymer_entity_metadata.to_csv(metadata_path/'polymer_entity_metadata.csv.gz', index=False)
    
    # entity instance / chain label
    print("Querying polymer entity instance / chain...")
    
    polymer_entity_instance_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_entity_instance, 
        folder_path=database_path,
        id_type='entry',
        j=j
    )
    polymer_entity_instance = pd.concat(polymer_entity_instance_list, axis=0)
    print(f"\tEntry coverage: {polymer_entity_instance['entry_id'].nunique() / num_pdb_entries}")
    polymer_entity_instance.to_csv(metadata_path/'polymer_entity_instance.csv.gz', index=False)

    # entity instance / chain annotation
    print('Querying annotations for polymer entity instance / chain...')
    polymer_entity_instance_metadata_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_polymer_entity_instance_annotation, 
        folder_path=database_path,
        id_type='entry',
        j=j
    )
    polymer_entity_instance_metadata = pd.concat(polymer_entity_instance_metadata_list)
    print(f"\tEntry coverage: {polymer_entity_instance_metadata['entry_id'].nunique() / num_pdb_entries:.2%}")
    polymer_entity_instance_metadata.to_csv(metadata_path/'polymer_entity_instance_annotation.csv.gz', index=False)

    # Interface info
    print('Querying polymer interface (first assembly only) metadata...')
    polymer_interface_metadata_list = rcsb_api.get_all_metadata(
        query_fn=rcsb_api.query_assembly_interface_metadata, 
        folder_path=database_path,
        id_type='first_assembly',
        j=j
    )
    polymer_interface_metadata = pd.concat(polymer_interface_metadata_list)
    print(f"\tEntry coverage: {polymer_interface_metadata['entry_id'].nunique() / num_pdb_entries:.2f}")
    polymer_interface_metadata.to_csv(metadata_path/'polymer_interface_metadata.csv.gz', index=False)


def main(args):
    if args.rsync_rcsb_database:
        rsync_rcsb(rcsb_root_path=args.database_path, dry_run=args.dry_run)
    
    if args.update_metadata:
        query_metadata(database_path=args.database_path, metadata_path=args.metadata_path)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Update RCSB PDB database",
        description="Update local copy of RCSB PDB database and query metadata, including:\n" + \
            "\t- entry_metadata.csv.gz: metadata for PDB entry\n" + \
            "\t- assembly_metadata.csv.gz: metadata for biological assemblies\n" + \
            "\t- polymer_entity_metadata.csv.gz: metadata for polymeric entities (unique protein or nucleic acid sequences)\n" + \
            "\t- polymer_entity_instance.csv.gz: metadata for polymeric entity instances (a.k.a chains)\n" + \
            "\t- polymer_interface_metadata.csv.gz: metadata for interfaces between polymer instances (only in the first assembly)\n" + \
            "\t- polymer_entity_instance_annotation.csv.gz: annotation records for polymer entity instances",
        formatter_class=RawDescriptionHelpFormatter
    )

    parser.add_argument("--database-path", type=str, default='RCSB_pdb/',
        help="Path to RCSB database root folder")
    parser.add_argument("--metadata-path", type=str, default='.',
        help="Path to RCSB metadata files")
    parser.add_argument("--rsync-rcsb-database", action='store_true',
        help="If rscync RCSB local copy with updates")
    parser.add_argument("--dry-run", action='store_true',
        help='If dry run rsync')
    parser.add_argument("--update-metadata", action='store_false',
        help="If update metadata (entry, assembly, entity, chain, interface)")
    parser.add_argument("-j", type=int, default=20,
        help="Number of parallel porocesses when querying the GraphQL database")

    args = parser.parse_args()
    main(args)


