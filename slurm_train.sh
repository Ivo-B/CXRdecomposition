#!/bin/bash

#SBATCH --partition=maxgpu,hzg,mdlma   # Maximum time request
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint="V100&GPUx4"
#SBATCH --mail-type ALL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user ivo.baltruschat@desy.de  # Email to which notifications will be sent
#SBATCH --export=NONE
#SBATCH --export=LD_PRELOAD=""
#SBATCH --output    /beegfs/desy/user/ibaltrus/slurm/TrainU-%N-%j.out  # File to which STDOUT will be written
#SBATCH --error     /beegfs/desy/user/ibaltrus/slurm/TrainU-%N-%j.err  # File to which STDERR will be written


source /etc/profile.d/modules.sh
source /home/ibaltrus/.bashrc

module load maxwell cuda/11.1
export LD_LIBRARY_PATH=/software/cuda/cuda-11.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
source activate py36-Tensorflow2

cd /beegfs/desy/user/ibaltrus/repos/cxrDecomp
#git tag -a $1 -m "slurm experiment"
python -u ./main_aritra.py $1





