#!/bin/bash
#SBATCH --job-name=family_matching_20M
#SBATCH -p normal # partition (queue) (normal, dev, gpu)
#SBATCH -N 1 # numberof nodes(entre 1 y 5) Solo si usa MPI
#SBATCH -n 12 # number of cores (max 32)
#SBATCH -t 5-00:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL #NONE, BEGIN, END, FAIL, REQUEUE, ALL (TIME_LIMIT_90 (reached 90 percent of time limit))
#SBATCH --mail-user= #EMAIL

module load python/3.6.6
python make_dataset.py