#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-15
#SBATCH -p hebbe
#SBATCH -J cv7
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -C MEM128
#SBATCH -t 10:00:00
#SBATCH -o cv_7.stdout
#SBATCH -e cv_7.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp sample_submission.csv.zip $TMPDIR
pdcp numeric_b1_b7_nf149.hdf $TMPDIR
pdcp count_encode.hdf $TMPDIR
pdcp benchmark_8_numeric_features_CV_7_count_encode.py $TMPDIR
pdcp bosch_helper.py $TMPDIR

cd $TMPDIR

python benchmark_8_numeric_features_CV_7_count_encode.py

cp *.pickle $SLURM_SUBMIT_DIR
cp *.gz $SLURM_SUBMIT_DIR
# End script