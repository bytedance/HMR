## Dataset generation pipeline for MaSIF-ligand ligand pocket classification task

Input data to this dataset generation pipeline can be downloaded from [Zenodo](https://zenodo.org/record/7686423/files/MasifLigand_auxdata.tar?download=1).

Use `python3 [scirpt.py] -h` to see additional options for each step.

### prepare input for MSMS
```bash
python3 step1_pdb_to_xyzrn.py \
    --MasifLigand-source [MaSIF-ligand PDB folder] \
    --out-root [MaSIF-ligand mesh data dir] \
    --pdb2pqr-bin [pdb2pqr binary path]
```

### execute MSMS to generate molecular surface
```bash
python3 step2_compute_msms.py  \
    --data-root [MaSIF-ligand mesh data dir] \
    --msms-bin ${MSMS_bin}/msms.x86_64Linux2.2.6.1 \
    --probe-radius 1.0 \
    --density 2.0
```

### extract ligand binding pocket
```bash
python3 step3_extract_pocket.py \
    --data-root [MaSIF-ligand mesh data dir] \
    --ligand-root [MaSIF-ligand ligand folder] \
    --resolution 0.5 \
    --radius 4.0
```

### compute Laplace-Beltrami basis and generate dataset
```bash
python3 step4_gen_dataset.py \
    --data-root [MaSIF-ligand mesh data dir] \
    --out-dir [dataset output dir] \
```


