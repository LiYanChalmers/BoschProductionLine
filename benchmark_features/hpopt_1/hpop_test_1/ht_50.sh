#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-15
#SBATCH -p hebbe
#SBATCH -J ht_50
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -C MEM128
#SBATCH -t 0-16:0:0
#SBATCH -o ht_50.stdout
#SBATCH -e ht_50.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp sample_submission.csv.zip $TMPDIR
pdcp numeric_b1_b8_nf149_1.hdf $TMPDIR
pdcp ht_50.py $TMPDIR
pdcp bosch_helper.py $TMPDIR

cd $TMPDIR

python ht_50.py

cp *.pickle $SLURM_SUBMIT_DIR
cp *.gz $SLURM_SUBMIT_DIR
# End script