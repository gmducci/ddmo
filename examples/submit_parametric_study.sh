#!/bin/bash -l

#SBATCH --job-name=goodwin-param
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH -o ./logs/tjob_%A_%a.out
#SBATCH -e ./logs/tjob_%A_%a.err

module purge

# Activate your environment
source /u/jmlombardi/DDMO/DDMO/bin/activate
module load python-waterboa/2025.06

# Create logs directory if it doesn't exist
mkdir -p logs

# Map SLURM array index to your code
JOB_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_JOBS=10

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: ${JOB_ID}"
echo "Total Jobs: ${TOTAL_JOBS}"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"
echo "=========================================="
echo

# Run the parametric study
# This will distribute 36 grid points across 10 jobs
# Each job processes 3-4 grid points
python3 ex_goodwin_parametric_study.py \
    --job-id ${JOB_ID} \
    --total-jobs ${TOTAL_JOBS} \
    --n-gen 20 \
    --pop-size 200 \
    --output-dir ./results_cluster

echo
echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
