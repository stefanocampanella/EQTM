#! /usr/bin/env bash
#SBATCH --job-name=RISE
#SBATCH --account=IscrB_PALEOMIT
#SBATCH --partition=m100_usr_prod
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%A_%a.log
#SBATCH --verbose

PROJECT_ROOT=$CINECA_SCRATCH/correlation-detector
DATA_ROOT=$CINECA_SCRATCH/RISE_DATA_MARCONI_REVISED/with_HN
TEMPLATES=$DATA_ROOT/templates
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
DAYS_PER_TASK=16

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PATH=$PROJECT_ROOT/bin:$PATH

mapfile -t SOURCE_BATCH < <(find "$SOURCE_DIR" -name "*.mseed" | sort -n | head -n $((DAYS_PER_TASK * (SLURM_ARRAY_TASK_ID + 1))) | tail -n $DAYS_PER_TASK)
for INDEX in "${!SOURCE_BATCH[@]}"
do
  SOURCE=${SOURCE_BATCH[INDEX]}
  export CUDA_VISIBLE_DEVICES=$((INDEX % 4)),
  mkdir -p "$OUTPUT_DIR"
  OUTPUT="$OUTPUT_DIR/$(basename "${SOURCE%.mseed}")"
  correlation-detector "$SOURCE" "$TEMPLATES/travel_times" "$TEMPLATES/data" "$OUTPUT" &
done
wait

exit