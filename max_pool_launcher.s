#!/bin/bash

#SBATCH --job-name=max_pool
#SBATCH --mem=10GB
#SBATCH --time=00:02:00
#SBATCH --output=max_pool.out
#SBATCH --gres=gpu:p40:1 –mem 20GB
/home/ppo208/lab3/max_pool
