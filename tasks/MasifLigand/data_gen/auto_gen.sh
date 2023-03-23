#!/bin/bash


raw_data=/root/Workdir/_ByteNAS/MaSIF-ligand/data/masif-ligand-dataset/
workdir=/root/Workdir/data-gen-2/MasifLigand_mesh/
dataset_dir=/root/Workdir/data-gen-2/dataset_MasifLigand
j=64

export PYTHONPATH=/root/Workdir/HMR_public/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/miniconda3/lib # some problem with conda installation

export PDB2PQR_BIN=/root/miniconda3/envs/HMR/bin/pdb2pqr30
export MSMS_BIN=/root/MSMS/msms.x86_64Linux2.2.6.1

mkdir -p $workdir

echo "STEP1"
python3 step1_pdb_to_xyzrn.py --MasifLigand-source $raw_data/pdb --out-root $workdir -j $j | tee $workdir/step1_output.log

echo "STEP2"
python3 step2_compute_ses.py --data-root $workdir -j $j | tee $workdir/step2_output.log

echo "STEP3"
python3 step3_extract_pocket.py --data-root $workdir --ligand-root $raw_data/ligand -j $j | tee $workdir/step3_output.log

echo "STEP4"
python3 step4_gen_dataset.py --data-root $workdir --out-dir $dataset_dir -j 2 | tee $workdir/step4_output.log
