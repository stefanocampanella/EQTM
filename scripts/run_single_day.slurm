#! /usr/bin/env bash
#SBATCH --job-name=RISE
#SBATCH --account=IscrB_PALEOMIT
#SBATCH --array=0-3300
#SBATCH --partition=m100_usr_preempt
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%A_%a.log
#SBATCH	--requeue 
#SBATCH --verbose

PROJECT_ROOT=$CINECA_SCRATCH/correlation-detector
DATA_ROOT=$CINECA_SCRATCH/RISE_DATA_MARCONI_REVISED/with_HN
TEMPLATES_DIR=$DATA_ROOT/templates
TEMPLATES_BATCHES_DIR=$DATA_ROOT/templates/batches
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs
BATCHES_OUTPUT_DIR=$TMPDIR/outputs
STATS_DIR=$DATA_ROOT/stats
CATALOGUE=$TEMPLATES_DIR/catalogue.zmap

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PATH=$PROJECT_ROOT/bin:$PATH

mapfile -t ALL_SOURCES < <(find "$SOURCE_DIR" -type f -name "*.mseed" | sort -n )
SOURCE="${ALL_SOURCES[SLURM_ARRAY_TASK_ID]}"
SOURCE_NAME=$(basename "${SOURCE%.mseed}")

mkdir -p "$BATCHES_OUTPUT_DIR"
mapfile -t TEMPLATES_BATCHES < <(find "$TEMPLATES_BATCHES_DIR" -type d -mindepth 1 -maxdepth 1 | sort -n)
for BATCH in "${!TEMPLATES_BATCHES[@]}"
do
  TEMPLATES_BATCH=${TEMPLATES_BATCHES[BATCH]}
  BATCH_OUTPUT=$BATCHES_OUTPUT_DIR/${SOURCE_NAME}_$BATCH
  export CUDA_VISIBLE_DEVICES=$((BATCH % 4)),
  correlation-detector "$SOURCE" "$TEMPLATES_BATCH/travel_times" "$TEMPLATES_BATCH/data" "$BATCH_OUTPUT" &
done
wait

mkdir -p "$OUTPUT_DIR"
OUTPUT=$OUTPUT_DIR/$SOURCE_NAME
cat-detections "$BATCHES_OUTPUT_DIR" "$OUTPUT"
mkdir -p "$STATS_DIR"
STATS=$STATS_DIR/$SOURCE_NAME
detections-stats "$CATALOGUE" "$OUTPUT" "$STATS"

exit
