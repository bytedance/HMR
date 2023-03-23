## Dataset generation pipeline for rigid protein-protein docking task.

Input data to this dataset generation pipeline can be downloaded from the RCSB database.

### sync with RCSB
```bash
rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/biounit/coordinates/divided/ RCSB_pdb
```

### obtain RCSB PDB metadata
```bash
python step1_query_rcsb_metadata.py
```

### filter metadata
```bash
python step2_filter_rcsb_metadata.py
```

### add Hydrogen atoms
```bash
python step3_pairs_to_pqr.py  --pdb2pqr-bin [/path/to/pdb2pqr/executable]
```

### convert pqr to xyzrn format
```bash
python step4_pqr_to_xyzrn.py
```

### execute MSMS to generate molecular surface
```bash
python step5_compute_ses.py 
```

### refine surface mesh
```bash
python step6_refine_mesh.py
```

### compute Laplace-Beltrami basis and generate dataset
```bash
python step7_gen_dataset.py 
```

