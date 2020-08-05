#!/bin/bash
# Job name:
#SBATCH --job-name=NN4UQ_ns_2
#
# Account:
#SBATCH --account=fc_mllam
#
# Partition:
#SBATCH --partition=savio2
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#
# Wall clock limit:
#SBATCH --time=08:00:00
#
# Email
#SBATCH --mail-type=END
#SBATCH --mail-user=schweini@berkeley.edu

## Command(s) to run:
python ../noah.py