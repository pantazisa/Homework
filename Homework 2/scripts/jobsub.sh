#!/bin/bash
#SBATCH --job-name=cc_cilk_mpi
#SBATCH --output=res_full.out
#SBATCH --error=err_full.err
#SBATCH --open-mode=append
#SBATCH --partition=rome              
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1        
#SBATCH --cpus-per-task=64      
#SBATCH --time=00:40:00
#SBATCH --mem=0

module purge
module load gcc openmpi      
module load OpenCilk    

# Set this to the path of your matrix file
MTX_FILE="your/path/to/matrix.mtx"

export MPICH_CC=clang
export OMPI_CC=clang
export CILK_NWORKERS=64
export CILK_WORKER_STACK_SIZE=32M

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "MPI Ranks per Node: $SLURM_NTASKS_PER_NODE"
echo "Cilk Workers per Rank: $CILK_NWORKERS"

if [ ! -f ./cc_cilk_mpi ]; then
    echo "Error: Executable ./cc_cilk_mpi not found!"
    exit 1
fi

srun ./cc_cilk_mpi "$MTX_FILE"