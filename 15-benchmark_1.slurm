#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -A optinetly6j
#SBATCH -o 15-benchmark_1.stdout
#SBATCH -e 15-benchmark_1.stderr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard
#SBATCH --mem=64G

module load anaconda3
source activate kaggle

python 15-benchmark_1.py