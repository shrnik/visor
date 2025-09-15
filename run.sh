#!/bin/bash

# set output and error output filenames, %j will be replaced by Slurm with the jobid
#SBATCH -o download%j.out
#SBATCH -e download%j.err

# single node in the "short" partition
#SBATCH -N 10
#SBATCH -p short

#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL

# half hour timelimit
#SBATCH -t 4:00:00

module load python3/3.12.9

cd visor
source .venv/bin/activate

python3 -m core.main download-worker