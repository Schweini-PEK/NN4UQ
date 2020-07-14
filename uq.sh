#!/bin/bash
# Job name:
#SBATCH --job-name=NN4UQ_test
#
# Account:
#SBATCH --account=fc_mllam
#
# Partition:
#SBATCH --partition=savio
#
# Quality of Service:
#SBATCH --qos=savio_normal
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#
# Wall clock limit:
#SBATCH --time=01:00:00
#
# Email
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=schweini@berkeley.edu

## Command(s) to run:
python uq_ray.py