#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -A optinetly6j
#SBATCH -o bosch_1.stdout
#SBATCH -e bosch_1.stderr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=standard

module load anaconda3
source activate kaggle

python cv_test.py