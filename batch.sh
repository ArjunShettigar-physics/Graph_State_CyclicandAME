#!/bin/sh
#PBS -N ame_gpu
#PBS -q gpu
#PBS -e out_gpu225.err
#PBS -o output_gpu225.out
#PBS -l walltime=200000:00:00
#PBS -l select=1:ncpus=1

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

. /etc/profile.d/modules.sh
module load python3/3.10.9
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
module load cuda/11.0.2
module load openmpi/4.1.4     # ← add this



export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpirun -np 2 python3 parallel_gpus.py