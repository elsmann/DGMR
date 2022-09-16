#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=20:00
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=frederike.elsmann@ru.nl

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/arch/Centos8/EB_production/2021/software/CUDA/11.3.1/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/arch/Centos8/EB_production/2021/software/CUDA/11.3.1/stubs/lib64
mkdir -p /scratch-shared/$USER
cp -r $HOME/data/TFR /scratch-shared/$USER

cd /scratch-shared/$USER

python $HOME/models/DGMR/train.py TFR 2009 


