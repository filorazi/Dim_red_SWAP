#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --begin=2021-07-10T02:00:00 # Defer start to this much time
#SBATCH --time=7-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH --mem-per-cpu=32000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --array=0-39
#SBATCH -o ./Logs/VAE_06_earth_mover_%A_%a.out # File to which STDOUT will be written
#SBATCH -e ./Logs/VAE_06_earth_mover_%A_%a.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=marton.kanasz-nagy@mpq.mpg.de # Email to which notifications will be sent

module purge
module load anaconda/3/2023.03

echo "Job runs on node:"
echo $SLURM_NODELIST
cd $SLURM_SUBMIT_DIR

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID

DATATYPES=(ising heisenberg)
COEFFTRACEDCOST=1
NUMCONVPERLAYERS=(0 1 2 4)
DATATYPE=${DATATYPES[$(( ${TASK_ID} % 8 % 2 ))]}
NUMINPUTQUBITS=8
NUMMIDDLEQUBITS=2
NUMPOOLINGLAYERS=2
NUMCONVPERLAYER=${NUMCONVPERLAYERS[$(( ${TASK_ID} % 8 / 2 ))]}
NUMOPTSTEPS=5000
NUMRESTARTS=20
NUMOPTSTEPSPERRESTARTS=$(( ${NUMOPTSTEPS} / ${NUMRESTARTS} ))
FRACSAMPLEDSTATES=0.25
LISTOPSUPPORT="1 2 3"
LISTOPSUPPORTPROBS="1. 1. 1."
LISTOPSUPPORTMAXRANGE="1 5 3"
RUNID=$(( ${TASK_ID} / 8 ))

echo DATATYPE=$DATATYPE
echo COSTTYPE=earth-mover
echo COEFFTRACEDCOST=$COEFFTRACEDCOST
echo NUMINPUTQUBITS=$NUMINPUTQUBITS
echo NUMMIDDLEQUBITS=$NUMMIDDLEQUBITS
echo NUMPOOLINGLAYERS=$NUMPOOLINGLAYERS
echo NUMCONVPERLAYER=$NUMCONVPERLAYER
echo NUMOPTSTEPS=$NUMOPTSTEPS
echo NUMRESTARTS=$NUMRESTARTS
echo FRACSAMPLEDSTATES=$FRACSAMPLEDSTATES
echo LISTOPSUPPORT=$LISTOPSUPPORT
echo LISTOPSUPPORTPROBS=$LISTOPSUPPORTPROBS
echo LISTOPSUPPORTMAXRANGE=$LISTOPSUPPORTMAXRANGE
echo RUNID=$RUNID


srun -N 1 -n 1 python vae_06.py --${DATATYPE} --cost-earth-mover --coeff-traced-cost=${COEFFTRACEDCOST} --num-input-qubits=${NUMINPUTQUBITS} --num-middle-qubits=${NUMMIDDLEQUBITS} --num-pooling-layers=${NUMPOOLINGLAYERS} --num-conv-per-layer=${NUMCONVPERLAYER} --num-optimization-steps=${NUMOPTSTEPSPERRESTARTS} --fraction-of-sampled-states=${FRACSAMPLEDSTATES} --same-compress-and-expand-params --even-odd-symmetry-breaking --list-op-support $LISTOPSUPPORT --list-op-support-probs $LISTOPSUPPORTPROBS --list-op-support-max-range $LISTOPSUPPORTMAXRANGE --op-translation-inveriance --save-last --run-id=$RUNID --no-load --no-save-opt-plot --no-save-opt-txt

for (( i=2; i<=(( $NUMRESTARTS - 1)); i++ ))
do
   echo ""
   echo ""
   echo ""
   echo "-------------------------"
   echo "Running restart no. $i"
   echo "-------------------------"
   srun -N 1 -n 1 python vae_06.py --${DATATYPE} --cost-earth-mover --coeff-traced-cost=${COEFFTRACEDCOST} --num-input-qubits=${NUMINPUTQUBITS} --num-middle-qubits=${NUMMIDDLEQUBITS} --num-pooling-layers=${NUMPOOLINGLAYERS} --num-conv-per-layer=${NUMCONVPERLAYER} --num-optimization-steps=${NUMOPTSTEPSPERRESTARTS} --fraction-of-sampled-states=${FRACSAMPLEDSTATES} --same-compress-and-expand-params --even-odd-symmetry-breaking --list-op-support $LISTOPSUPPORT --list-op-support-probs $LISTOPSUPPORTPROBS --list-op-support-max-range $LISTOPSUPPORTMAXRANGE --op-translation-inveriance --save-last --no-save-opt-plot --no-save-opt-txt --run-id=$RUNID
done

srun -N 1 -n 1 python vae_06.py --${DATATYPE} --cost-earth-mover --coeff-traced-cost=${COEFFTRACEDCOST} --num-input-qubits=${NUMINPUTQUBITS} --num-middle-qubits=${NUMMIDDLEQUBITS} --num-pooling-layers=${NUMPOOLINGLAYERS} --num-conv-per-layer=${NUMCONVPERLAYER} --num-optimization-steps=${NUMOPTSTEPSPERRESTARTS} --fraction-of-sampled-states=${FRACSAMPLEDSTATES} --same-compress-and-expand-params --even-odd-symmetry-breaking --list-op-support $LISTOPSUPPORT --list-op-support-probs $LISTOPSUPPORTPROBS --list-op-support-max-range $LISTOPSUPPORTMAXRANGE --op-translation-inveriance --save-best --no-save-opt-plot --no-save-opt-txt --run-id=$RUNID


echo "job finished"
wait

