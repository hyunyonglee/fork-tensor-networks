#!/bin/bash
#SBATCH --job-name=TEST                  # Job name
#SBATCH --cpus-per-task=48           # Number of cores per MPI task 
#SBATCH --ntasks=1                   # Number of MPI tasks (i.e. processes)
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --ntasks-per-node=1          # Maximum number of tasks on each node.
#SBATCH --time=2-24:00:00              # Wall time limit (days-hrs:min:sec)
#SBATCH --output=../Out/TEST.out             # Path to the standard output and error files relative to the working directory

export JULIA_NUM_THREADS=24

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
srun --mpi=pmi2 /home/dinnercha/ITensor/Jobs/FTN/Work/Mywork/Lauch.jl
