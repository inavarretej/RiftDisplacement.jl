#!/bin/bash      
#SBATCH --output=/home/navarrete/JustRift.jl/output/Rift2D_velocity.%j.out
#SBATCH --error=/home/navarrete/JustRift.jl/output/Rift2D_velocity.%j.err                                          
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=vel
#SBATCH --partition=geo4dgpu
#SBATCH --mem=0
#SBATCH --time=unlimited    
#SBATCH --gpus-per-node=1

# Print stuff                                               
echo Working directory is $SLURM_SUBMIT_DIR                                 
cd $SLURM_SUBMIT_DIR
echo Running on host `hostname`
echo Time is `date`
echo This job has allocated $SLURM_JOB_NUM_NODES nodes                           
# Tell me which nodes it is run on                                     

echo " "                                                  
echo This jobs runs on the following processors:                              
echo $SLURM_JOB_NODELIST                                          
echo " "                                                  

export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_CUDAAWARE_MPI=1 # IGG
export JULIA_CUDA_USE_COMPAT=false # IGG

# Run the simulations
module load julia

export PATH="/share/apps/julia/bin:$PATH"

julia --project=.. -O3  -t 2 Rift2D_velocity.jl


echo Time is `date`
