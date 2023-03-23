## Protein-protein rigid docking task

### Dataset generation
Docking Benchmark 5.5 Data can be downloaded from [Zenodo](https://zenodo.org/record/7686423/files/Docking_auxdata.tar?download=1).
RCSB training data should be manually generated, see `data_gen` for details. 

### Make protein complex structure prediction on the Docking Benchmark 5.5 dataset
```bash
pred_args=(
    --config config.yaml
    --data_dir [path/to/Docking_auxdata/]
    --mute_tqdm False
    --fp16 True
    --restore [path/to/Docking_auxdata/trained_docking_model.pt]
)

# protein binding site prediction
python predict.py "${pred_args[@]}"

# protein docking
python docking.py "${pred_args[@]}"
```
 
### Train the model from scratch 
You should generate the RCSB dataset first. Please see `data_gen` for details. 
After generating the RCSB training dataset, put dataset_RCSB and dataset_DB5 (can be found in `Docking_auxdata` folder) under the same dataset directory.
We also provide a small toy training set (`Docking_auxdata/dataset_RCSB`) so that users can run the training code.
    
```bash
train_args=(
    --config config.yaml
    --data_dir [/path/to/dataset/]
    --mute_tqdm False
    --fp16 True
)

# Distributed training using Horovod
horovodrun -np [num_GPUs] python main.py "${train_args[@]}"

# OR single-GPU training
python main.py "${train_args[@]}"
```
