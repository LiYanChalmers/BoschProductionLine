#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-15
#SBATCH -p hebbe
#SBATCH -J ht_1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -C MEM64
#SBATCH -t 0-0:10:0
#SBATCH -o hpopt_test_1.stdout
#SBATCH -e hpopt_test_1.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp sample_submission.csv.zip $TMPDIR
pdcp numeric_b1_b7_nf149.hdf $TMPDIR
pdcp hpopt_test_1.py $TMPDIR
pdcp bosch_helper.py $TMPDIR

cd $TMPDIR

python hpopt_test_1.py

cp *.pickle $SLURM_SUBMIT_DIR
cp *.gz $SLURM_SUBMIT_DIR
# End script