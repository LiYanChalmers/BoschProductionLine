#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-15
#SBATCH -p hebbe
#SBATCH -J hpopt_test_0
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -C MEM128
#SBATCH -t 00:10:00
#SBATCH -o hpopt_test_0.stdout
#SBATCH -e hpopt_test_0.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp sample_submission.csv.zip $TMPDIR
pdcp numeric_b1_b7_nf149.hdf $TMPDIR
pdcp hpopt_test_0.py $TMPDIR
pdcp bosch_helper.py $TMPDIR

cd $TMPDIR

python hpopt_test_0.py

cp *.pickle $SLURM_SUBMIT_DIR
cp *.gz $SLURM_SUBMIT_DIR
# End script