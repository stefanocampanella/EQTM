#! /usr/bin/env bash
#SBATCH --job-name=RISE
#SBATCH --account=IscrB_PALEOMIT
#SBATCH --array=0-165
#SBATCH --partition=m100_usr_prod
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue
#SBATCH --verbose

PROJECT_ROOT=$CINECA_SCRATCH/correlation-detector
DATA_ROOT=$CINECA_SCRATCH/RISE_DATA_MARCONI_REVISED/with_HN
TEMPLATES=$DATA_ROOT/templates
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
STATS_DIR=$DATA_ROOT/stats
CATALOGUE=$TEMPLATES_DIR/catalogue.zmap
DAYS_PER_TASK=20

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PATH=$PROJECT_ROOT/bin:$PATH

mkdir -p "$OUTPUT_DIR"
mkdir -p "$STATS_DIR"
mapfile -t SOURCE_BATCH < <(find "$SOURCE_DIR" -type f -name "*.mseed" | sort -n | head -n $((DAYS_PER_TASK * (SLURM_ARRAY_TASK_ID + 1))) | tail -n $DAYS_PER_TASK)
for INDEX in "${!SOURCE_BATCH[@]}"
do
  SOURCE=${SOURCE_BATCH[INDEX]}
  SOURCE_NAME=$(basename "${SOURCE%.mseed}")
  OUTPUT=$OUTPUT_DIR/$SOURCE_NAME
  STATS=$STATS_DIR/$SOURCE_NAME
  export CUDA_VISIBLE_DEVICES=$((INDEX % 4)),
  correlation-detector "$SOURCE" "$TEMPLATES/travel_times" "$TEMPLATES/data" "$OUTPUT" && detections-stats "$CATALOGUE" "$OUTPUT" "$STATS" &
done
wait


exit