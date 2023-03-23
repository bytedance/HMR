## QM9 molecular property regression

We use the same train-valid-test split as that in "E(n) Equivariant Graph Neural Networks" (Satorras et al., 2021).

To reproduce HMR model results:

### Dataset generation
Please see `data_gen` for details. Raw data can be downloaded from [Zenodo](https://zenodo.org/record/7686423).

### Model training
```bash
args=(
    --config config.json
    --data_dir [/path/to/generated/QM9/dataset/]
    --fp16 False
)

# Distributed training using Horovod
horovodrun -np [num_GPUs] python main.py "${args[@]}"

# OR single-GPU training
python main.py "${args[@]}"
```
