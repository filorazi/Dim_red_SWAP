#!/bin/bash -l

#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
# #SBATCH --begin=2021-07-10T02:00:00 # Defer start to this much time
#SBATCH --time=7-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem-per-cpu=32000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --array=0-21
#SBATCH -o ./Logs/ed/VAE_06_earth_mover_%A_%a.out # File to which STDOUT will be written
#SBATCH -e ./Logs/ed/VAE_06_earth_mover_%A_%a.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=filippo.orazi2@unibo.it # Email to which notifications will be sent


module purge
module load stack/2024-06 python/3.11.6

echo "Job runs on node:"
echo $SLURM_NODELIST
cd $SLURM_SUBMIT_DIR

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID

REPETITIONS=8
NUMINPUTQUBITS=8
NUMMIDDLEQUBITS=(1 2 3 4 5 6 7)
NUMMIDDLEQUBIT=${NUMMIDDLEQUBITS[$(( ${TASK_ID} % 3))]}
BATCHSIZES=(8 25 100)
BATCHSIZE=${BATCHSIZES[$(( ${TASK_ID} % 5 ))]}
EPOCHS=300
FRACSAMPLEDSTATES=0.25
LISTOPSUPPORT="1 2 3"
LISTOPSUPPORTPROBS="1. 1. 1."
LISTOPSUPPORTMAXRANGE="1 5 3"
RUNID=$(( ${TASK_ID} / 8 ))
VALSPLIT=0.2
OPTSTEP=0.2

echo REPETITIONS=$REPETITIONS
echo COSTTYPE=earth-mover
echo NUMINPUTQUBITS=$NUMINPUTQUBITS
echo NUMMIDDLEQUBIT=$NUMMIDDLEQUBIT
echo EPOCHS=$EPOCHS
echo OPTSTEP=$OPTSTEP
echo BATCHSIZE=$BATCHSIZE
echo VALSPLIT=$VALSPLIT
echo FRACSAMPLEDSTATES=$FRACSAMPLEDSTATES
echo LISTOPSUPPORT=$LISTOPSUPPORT
echo LISTOPSUPPORTPROBS=$LISTOPSUPPORTPROBS
echo LISTOPSUPPORTMAXRANGE=$LISTOPSUPPORTMAXRANGE
echo RUNID=$RUNID
echo 'NO JAX' 

echo "srun -N 1 -n 1 python Cluster/single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of 'runs/ed' -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx"

srun -N 1 -n 1 python Cluster_ed/single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of 'runs/ed' -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx
for (( i=2; i<=(( $NUMRESTARTS - 1)); i++ ))
do
   echo ""
   echo ""
   echo ""
   echo "-------------------------"
   echo "Running restart no. $i"
   echo "-------------------------"
   srun -N 1 -n 1 python Cluster_ed/single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of 'runs/ed' -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx
done 
srun -N 1 -n 1 python Clusterm_ed/single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of 'runs/ed' -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx

echo "job finished"
wait
