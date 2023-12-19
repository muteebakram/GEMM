#!/bin/bash -x
#SBATCH -M lonepeak
#SBATCH --account=owner-guest
#SBATCH --partition=lonepeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C c20
#SBATCH -c 20
#SBATCH --exclusive
#SBATCH -t 0:45:00
#SBATCH --exclude lp[076-082]

echo "*** Assigned Lonepeak Node: " $SLURMD_NODENAME | tee -a lonepeak_mm4_cpu.$SLURM_JOB_ID\.log
echo " "
./openmp.sh | tee -a lonepeak_mm4_cpu.$SLURM_JOB_ID\.log
