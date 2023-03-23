## `MaSIF-ligand` binding pocket classification

This binding pocket classification task is introduced in “Deciphering Interaction Fingerprints from Protein Molecular Surfaces Using Geometric Deep Learning.” (Gainza, et al, Nature Methods, 2020), called `MaSIF-ligand`.

Data splits are from Gainza, et al. and can be found under `splits`.

### Dataset generation
Data can be downloaded from [Zenodo](https://zenodo.org/record/7686423).
Please see `data_gen` for details. 


### Model training
```bash
# please see `python main.py -h` and config.json for a complete set of options
args=(
    --config config.json
    --run_name [run_name]
    --data_dir [/path/to/prepared_dataset]
    --processed_dir [/path/to/cache/processed_data]
    --train_split_file [/path/to/train-list.txt]
    --valid_split_file [/path/to/val-list.txt]
    --test_split_file [/path/to/test-list.txt]
    --out_dir [/path/to/save/model/chem_geom]
    --use_chem_feat [True/False]
    --use_geom_feat [True/False]
)

# Distributed training using Horovod
horovodrun -np [num_GPUs] python main.py "${args[@]}"

# OR single-GPU training
python main.py "${args[@]}"
```

### Prediction and evaluation
   Trained model can be found under the `checkpoints` folder of the dataset archive. 
   `[chem/geom/chem_geom]_best.pt` are the checkpoints for models using chemical, geometric, or both features. To evaluate model performance:
```bash
args=(
    --config config.json
    --use_chem_feat [True/False]
    --use_geom_feat [True/False]
    --model_dir [/path/to/saved/model/checkpoints.pt]
    --data_dir [/path/to/prepared_dataset]
    --processed_dir [/path/to/cache/processed_data]
    --test_list [/path/to/test-list.txt]
)

python predict.py "${args[@]}"
```