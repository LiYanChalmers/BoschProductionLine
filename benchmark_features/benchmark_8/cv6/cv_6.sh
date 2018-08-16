#!/usr/bin/env bash
#SBATCH -A C3SE2018-1-15
#SBATCH -p hebbe
#SBATCH -J cv6
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -C MEM64
#SBATCH -t 00:10:00
#SBATCH -o cv_6.stdout
#SBATCH -e cv_6.stderr
module purge 

export PATH="/c3se/NOBACKUP/users/lyaa/conda_dir/miniconda/bin:$PATH"
source activate kaggle

pdcp sample_submission.csv.zip $TMPDIR
pdcp numeric_b1_b7_nf149.hdf $TMPDIR
pdcp time_station.hdf $TMPDIR
pdcp benchmark_8_numeric_features_CV_6_stations.py $TMPDIR
pdcp bosch_helper.py $TMPDIR

cd $TMPDIR

python benchmark_8_numeric_features_CV_6_stations.py

cp *.pickle $SLURM_SUBMIT_DIR
cp *.gz $SLURM_SUBMIT_DIR
# End script