#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=15:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=frederike.elsmann@ru.nl

module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1


cd $HOME/data/radar_data

# create TFR record files from 1.1.2009 to 31.12.2009 
# from radar_data (directory with yearly directory files)
# t0 TFRecord directory 
python $HOME/models/write_TFR.py 2009 1 1 2009 12 31 radar_data $HOME/data/TFR_radar

