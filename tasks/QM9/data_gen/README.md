
## Dataset generation pipeline for QM9 molecular property regression task

Input data to this dataset generation pipeline can be downloaded from [Zenodo](https://zenodo.org/record/7686423/files/QM9_auxdata.tar?download=1).

### prepare input for MSMS
```bash
python step1_convert_npz_to_xyzrn.py --qm9-source [path/to/QM9_auxdata/]
```


### execute MSMS to generate molecular surface
```bash
python step2_compute_msms.py --msms-bin [path/to/MSMS/dir]/msms.x86_64Linux2.2.6.1
```

### refine surface mesh
```bash
python step3_refine_mesh.py
```

### compute Laplace-Beltrami basis and generate dataset
```bash
python step4_gen_dataset.py --out-dir [QM9_dataset/dirname]
```

### copy id_prop.npy file into your dataset folder, i.e.,
```bash
cp [path/to/QM9_auxdata]/id_prop.npy [QM9_dataset/dirname]
```

