#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-a10-*
#$ -l h=qa-rtx6k-*
#$ -e errors/
#$ -N sophia-50-epochs

# Required modules
module load conda
conda init bash
source activate sam

python sophia_code.py