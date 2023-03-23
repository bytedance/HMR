"""Submodule for querying the GraphQL database of RCSB

NOTE: only selected types of information are retrieved, see below for all data domains

More details:
    - RCSB Data API: https://data.rcsb.org/#data-api
    - Data attributes: https://data.rcsb.org/data-attributes.html
    - In-browser GraphQL tool: https://data.rcsb.org/graphql/index.html
    - RESTful API doc (similar data structure as GraphQL): https://data.rcsb.org/redoc/index.html


----
Copyright (2023) Bytedance Ltd. and/or its affiliates

This work is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.

"""

import requests
import numpy as np
import pandas as pd
from pathlib import Path
from helpers import mp_run


GraphQL_BASE_URL = "https://data.rcsb.org/graphql?query="


def base_query(ids, query_template, captalize=True, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Base GraphQL query function
    
    Args:
        ids (list): list of entry/assembly/etc ids. Will be formated as ["{id1}", "{id2}", ...]
        query_template (str): template string with '{}' to locate formatted ids list
        captalize (bool): if captalize ids
        GraphQL_BASE_URL (str): RCSB GraphQL base url
        retry (int): maximum number of retries if query fails
    
    Returns:
        dict: content in the 'data' domain of the JSON response; or None if failed
    """

    if retry is None or retry is False or retry == 0:
        retry = 1
    if captalize:
        ids = [id_.upper() for id_ in ids]
    res = None
    for _ in range(retry):
        try:
            query = query_template.format('"' + '", "'.join(ids) + '"')
            res = requests.get(GraphQL_BASE_URL + query)
            if res.status_code == 200:
                return res.json()['data']
            else:
                raise requests.HTTPError()
        except:
            pass
    
    # query fail
    print(f'Failed for {ids[0]}, ...: {res.json()}', flush=True)
    return None


def get_entry_id_blocks(folder_path, block_size=200):
    """Search RCSB subfolders to get all entry_ids and split them into blocks"""
    folder_path = Path(folder_path)
    entry_ids = list(set([
        f.name.strip('.gz').strip('.pdb').split('.')[0].upper() 
        for f in folder_path.glob('*/*')
    ]))
    entry_blocks = [
        entry_ids[start: min(start + block_size, len(entry_ids))] 
        for start in range(0, len(entry_ids), block_size)
    ]
    return entry_blocks


def get_assembly_id_blocks(folder_path, block_size=200):
    """Get [all] assembly ids and split into blocks"""
    folder_path = Path(folder_path)

    assembly_ids = list(set([
        f.name.strip('.gz').strip('.pdb').split('.')[0].upper() + '-' \
            + f.name.strip('.gz').strip('.pdb').split('.')[1].strip('pdb')
        for f in folder_path.glob('*/*')
    ]))
    assembly_blocks = [
        assembly_ids[start: min(start + block_size, len(assembly_ids))] 
        for start in range(0, len(assembly_ids), block_size)
    ]
    return assembly_blocks


def get_first_assembly_id_blocks(folder_path, block_size=200):
    """Get the first assembly '{ENTRY_ID}-1'"""
    entry_id_blocks = get_entry_id_blocks(folder_path, block_size=block_size)
    return [[eid + '-1' for eid in block] for block in entry_id_blocks]


def get_all_metadata(query_fn, folder_path, id_type, block_size=200, j=None):
    """Query the metadata for all the entry/assembly ids under folder_path/subfolder

    Return:
        list of results for each block
    """
    if id_type in ['entry', 'entry_id', 'pdb', 'pdb_id']:
        id_blocks = get_entry_id_blocks(folder_path, block_size=block_size)
    elif id_type in ['first_assembly', 'first_assembly_id']:
        id_blocks = get_first_assembly_id_blocks(folder_path, block_size=block_size)
    elif id_type in ['assembly', 'assembly_ids']:
        id_blocks = get_assembly_id_blocks(folder_path, block_size=block_size)
    else:
        raise ValueError("'id_type' should be one of 'entry', 'assembly', or 'first_assembly'")

    res_df_list = mp_run(query_fn, id_blocks, j=j)
    return res_df_list


entry_metadata_col_dtype = {
    "year": "Int16",
    "assembly_count": "Int16",
    "entity_count": "Int16",
    "polymer_entity_count": "Int16",
    "nonpolymer_entity_count": "Int16",
    "deposited_model_count": "Int16",
    "deposited_polymer_entity_instance_count": "Int16",
    "inter_mol_covalent_bond_count": "Int16",
    "inter_mol_metalic_bond_count": "Int16"
}


def query_entry_metadata(entry_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get selected PDB entry metadata for provided entry_ids

    Returns:
        pd.DataFrame: a metadata table, each row is an entry
    """
    
    query_template = """{{
        entries(entry_ids:[{}]) {{
            rcsb_entry_container_identifiers {{
                entry_id
            }}
            rcsb_entry_info {{
                assembly_count
                entity_count
                polymer_entity_count
                nonpolymer_entity_count
                polymer_composition
                deposited_model_count
                deposited_polymer_entity_instance_count
                experimental_method
                molecular_weight
                inter_mol_covalent_bond_count
                inter_mol_metalic_bond_count
                resolution_combined
            }}
            struct {{
                title
                pdbx_descriptor
            }}
            struct_keywords {{
                pdbx_keywords
                text
            }}
            exptl {{
                method
            }}
            citation {{
                year
            }}
        }}
    }}
    """

    res_json = base_query(entry_ids, query_template=query_template, captalize=True, 
                          GraphQL_BASE_URL=GraphQL_BASE_URL, retry=retry)['entries']

    def entry_json_to_series(entry_json):
        formated = {
            **{'entry_id': entry_json['rcsb_entry_container_identifiers']['entry_id']},
            **entry_json['rcsb_entry_info'],
            **entry_json.get('struct', {}),
            **entry_json.get('struct_keywords', {}),
            **{'exptl_method': ','.join(exptl['method'] for exptl in entry_json['exptl']) \
                if entry_json['exptl'] is not None else None}
        }
        if entry_json['citation'] is not None:
            formated['year'] = entry_json['citation'][0]['year']  # first
        if formated['resolution_combined'] is not None:
            formated['resolution_combined'] = np.mean([float(r) for r in formated['resolution_combined'] if r is not None])
        return pd.Series(formated)

    return pd.DataFrame([entry_json_to_series(entry_json) for entry_json in res_json])


assembly_metadata_col_dtype = {
    "polymer_entity_count": "Int16",
    "nonpolymer_entity_count": "Int16",
    "oligomeric_count": "Int16",
    "polymer_entity_count_protein": "Int16",
    "nonpolymer_entity_count": "Int16",
    "polymer_entity_instance_count": "Int16",
    "polymer_entity_instance_count_protein": "Int16",
    "nonpolymer_entity_instance_count": "Int16",
    "num_interfaces": "Int16",
    "num_interface_entities": "Int16",
    "num_homomeric_interface_entities": "Int16",
    "num_heteromeric_interface_entities": "Int16",
    "num_isologous_interface_entities": "Int16",
    "num_heterologous_interface_entities": "Int16",
    "num_protein_interface_entities": "Int16",
    "total_number_interface_residues": "Int32"
}


def query_assembly_metadata(entry_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get selected PDB assembly metadata
    
    Returns:
        pd.DataFrame: a metadata table, each row is an assembly
    """

    query_template = """{{
        entries(entry_ids:[{}]) {{
            assemblies {{
                rcsb_assembly_container_identifiers {{
                    rcsb_id
                    entry_id
                    assembly_id  
                }}
                pdbx_struct_assembly {{
                    details
                    method_details
                    oligomeric_count
                    oligomeric_details
                    rcsb_candidate_assembly
                    rcsb_details
                }}
                rcsb_assembly_info {{
                    polymer_composition
                    polymer_entity_count
                    polymer_entity_count_protein
                    nonpolymer_entity_count
                    
                    polymer_entity_instance_count
                    polymer_entity_instance_count_protein
                    nonpolymer_entity_instance_count
                    
                    selected_polymer_entity_types
                    
                    num_interfaces
                    num_interface_entities
                    num_homomeric_interface_entities
                    num_heteromeric_interface_entities
                    num_isologous_interface_entities
                    num_heterologous_interface_entities
                    num_protein_interface_entities
                    
                    total_assembly_buried_surface_area
                    total_number_interface_residues
                }}
                rcsb_struct_symmetry {{
                    oligomeric_state
                }} 
            }}
        }}
    }}"""

    res_json = base_query(entry_ids, query_template=query_template, captalize=True, 
                          GraphQL_BASE_URL=GraphQL_BASE_URL, retry=retry)['entries']

    def assembly_json_to_series(assembly_json):
        formated = {
            **assembly_json['rcsb_assembly_container_identifiers'],
            **assembly_json['pdbx_struct_assembly'],
            **assembly_json['rcsb_assembly_info'],
        }
        if assembly_json['rcsb_struct_symmetry'] is not None and \
            assembly_json['rcsb_struct_symmetry'][0]['oligomeric_state'] is not None:
            formated['oligomeric_state'] = ', '.join([sym['oligomeric_state'] for sym in assembly_json['rcsb_struct_symmetry']])
        return pd.Series(formated)

    return pd.DataFrame([assembly_json_to_series(assembly_json) \
        for entry_json in res_json \
        for assembly_json in entry_json['assemblies']
    ])


polymer_entity_col_dtype = {
    "rcsb_sample_sequence_length": "Int32",
    "identity_100": "Int32",
    "identity_95": "Int32",
    "identity_90": "Int32",
    "identity_70": "Int32",
    "identity_50": "Int32",
    "identity_30": "Int32",
}


def query_polymer_entity_metadata(entry_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get selected polymer entity metadata

    Returns:
        pd.DataFrame: a metadata table, each row is a polymer entity
    """

    query_template = """{{
        entries(entry_ids:[{}]) {{
            
            polymer_entities {{
                rcsb_polymer_entity_container_identifiers {{
                    entry_id
                    entity_id
                }}
                
                rcsb_cluster_membership {{
                    cluster_id
                    identity
                }}
                
                entity_poly {{
                    rcsb_sample_sequence_length
                    rcsb_entity_polymer_type
                    pdbx_strand_id
                }}
            }}
        }}
    }}
    """

    res_json = base_query(entry_ids, query_template=query_template, captalize=True, 
                          GraphQL_BASE_URL=GraphQL_BASE_URL, retry=retry)['entries']

    def entity_json_to_series(entity_json):
        formated = {
            **entity_json['rcsb_polymer_entity_container_identifiers'],
            **entity_json['entity_poly']
        }
        if entity_json['rcsb_cluster_membership'] is not None:
            formated.update({
                f"identity_{cluster['identity']}": cluster['cluster_id'] 
                for cluster in entity_json['rcsb_cluster_membership']
            })
        
        return pd.Series(formated)
    
    entity_list = []
    for entry_json in res_json:
        if entry_json['polymer_entities'] is not None:
            entity_list += [entity_json_to_series(entity_json) for entity_json in entry_json['polymer_entities']]

    return pd.DataFrame(entity_list)


def query_entity_instance(entry_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get the polymer entity chain mapping information

    Returns:
        pd.DataFrame: a mapping table, each row is a entry-entity-chain
    """

    query_template = """{{
        entries(entry_ids: [{}]) {{
            polymer_entities{{
                entity_poly {{
                    rcsb_entity_polymer_type
                    rcsb_sample_sequence_length
                }}
                polymer_entity_instances {{
                    rcsb_polymer_entity_instance_container_identifiers {{
                        entry_id
                        entity_id
                        asym_id
                        auth_asym_id
                    }}
                }}
            }}
        }}
    }}
    """
    
    res_json = base_query(entry_ids, query_template=query_template, captalize=True, 
                          GraphQL_BASE_URL=GraphQL_BASE_URL, retry=retry)['entries']
    
    all_records = []

    def entity_json_to_records(entity_json):
        entity_info = entity_json['entity_poly']
        if entity_json['polymer_entity_instances'] is not None:
            for entity_instance_json in entity_json['polymer_entity_instances']:
                entity_instance_info = entity_instance_json['rcsb_polymer_entity_instance_container_identifiers']
                all_records.append({
                    **entity_instance_info,
                    **entity_info,
                })

    for entry_json in res_json:
        if entry_json['polymer_entities'] is not None:
            [entity_json_to_records(entity_json) for entity_json in entry_json['polymer_entities']]

    return pd.DataFrame(all_records)


def query_polymer_entity_instance_annotation(entry_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get selected polymer entity instance/chain annotations

    Returns:
        pd.DataFrame: a metadata table, each row is an annotation of polymer entity instance
    """

    query_template = """{{
        entries(entry_ids:[{}]) {{
            polymer_entities {{
                entity_poly {{
                    rcsb_sample_sequence_length
                    rcsb_entity_polymer_type
                }}
            
                polymer_entity_instances {{
                    rcsb_polymer_entity_instance_container_identifiers {{
                        entry_id
                        entity_id
                        asym_id
                        auth_asym_id
                    }}
                
                    rcsb_polymer_instance_annotation {{
                        annotation_id
                        assignment_version
                        description
                        name
                        ordinal
                        type
                        provenance_source
                        annotation_lineage {{
                            depth
                            id
                            name
                        }}
                    }}
                }}
            }}
        }}
    }}"""
    
    res_json = base_query(entry_ids, query_template=query_template, captalize=True, 
                          GraphQL_BASE_URL=GraphQL_BASE_URL, retry=retry)['entries']
    
    all_annotation_records = []

    def entity_json_to_records(entity_json):
        entity_info = entity_json['entity_poly']
        if entity_json['polymer_entity_instances'] is not None:
            for entity_instance_json in entity_json['polymer_entity_instances']:
                entity_instance_info = entity_instance_json['rcsb_polymer_entity_instance_container_identifiers']

                if entity_instance_json['rcsb_polymer_instance_annotation'] is not None:
                    for annotation_json in entity_instance_json['rcsb_polymer_instance_annotation']:
                        annotation_info = {
                            'annotation_type': annotation_json['type'],
                            'annotation_source': annotation_json['provenance_source'],
                            'annotation_ordinal': annotation_json['ordinal'],
                            'annotation_name': annotation_json['name'],
                            'annotation_description': annotation_json['description']
                        }

                        annotation_info.update({
                            f"annotation_lineage_{lineage['depth']}": f"{lineage['id']}|{lineage['name']}" 
                            for lineage in annotation_json['annotation_lineage']
                        })

                        annotation_info['annotation_id'] = annotation_json['annotation_id']
                        annotation_info['annotation_version'] = annotation_json['assignment_version']
                        all_annotation_records.append({
                            **entity_instance_info,
                            **entity_info,
                            **annotation_info
                        })

    
    for entry_json in res_json:
        if entry_json['polymer_entities'] is not None:
            [entity_json_to_records(entity_json) for entity_json in entry_json['polymer_entities']]

    return pd.DataFrame(all_annotation_records)


def query_assembly_interface_metadata(assembly_ids, GraphQL_BASE_URL=GraphQL_BASE_URL, retry=5):
    """Get selected interface metadata between entity instances (chain/asym)

    Returns:
        pd.DataFrame: a metadata table, each row is an interface
    """

    query_template = """{{
        assemblies(assembly_ids:[{}]) {{
            rcsb_assembly_info {{
                entry_id
                assembly_id
                polymer_entity_count_protein
                polymer_entity_instance_count
                nonpolymer_entity_instance_count
                num_interfaces
                num_interface_entities
                num_homomeric_interface_entities
                num_heteromeric_interface_entities
                num_isologous_interface_entities
                num_heterologous_interface_entities
                num_protein_interface_entities
                num_na_interface_entities
                num_prot_na_interface_entities
                total_assembly_buried_surface_area
                total_number_interface_residues
            }}
            interfaces {{
                rcsb_interface_container_identifiers {{
                    interface_id
                    interface_entity_id
                    rcsb_id
                }}
                rcsb_interface_info {{
                    polymer_composition
                    interface_character
                    interface_area
                    num_interface_residues
                    num_core_interface_residues
                }}
                rcsb_interface_partner {{
                    interface_partner_identifier {{
                        entity_id
                        asym_id
                    }}
                }}
            }}
        }}
    }}"""

    res_json = base_query(assembly_ids, query_template=query_template, captalize=True)

    def format_assem_interface_info(assem_json):
        assem_info = assem_json['rcsb_assembly_info']
        if assem_info['num_interfaces'] is not None and assem_info['num_interfaces'] > 0:
            interface_list = assem_json['interfaces']
            interface_info_list = [
                {
                    **interface_json['rcsb_interface_container_identifiers'],
                    **interface_json['rcsb_interface_info'],
                    **{
                        "chain_1": interface_json['rcsb_interface_partner'][0]['interface_partner_identifier']['asym_id'],
                        "chain_2": interface_json['rcsb_interface_partner'][1]['interface_partner_identifier']['asym_id']
                        }
                } for interface_json in interface_list
            ]
            return pd.concat([pd.DataFrame([assem_info] * len(interface_info_list)), pd.DataFrame(interface_info_list)], axis=1)
        else:
            return None
    
    if 'assemblies' in res_json.keys():
        res_df_list = [format_assem_interface_info(assem_json) for assem_json in res_json['assemblies']]
        try:
            return pd.concat(res_df_list, axis=0).reset_index(drop=True)
        except:
            return None
    else:
        return 'error'


