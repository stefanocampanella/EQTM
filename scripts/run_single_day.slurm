#! /usr/bin/env bash
#SBATCH --job-name=RISE
#SBATCH --account=IscrB_PALEOMIT
#SBATCH --partition=m100_usr_preempt
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%A_%a.log
#SBATCH --verbose

PROJECT_ROOT=$CINECA_SCRATCH/correlation-detector
DATA_ROOT=$CINECA_SCRATCH/RISE_DATA_MARCONI_REVISED/with_HN
TEMPLATES_BATCHES_DIR=$DATA_ROOT/templates/batches
SOURCE_DIR=$DATA_ROOT/data
OUTPUT_DIR=$DATA_ROOT/outputs

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export PATH=$PROJECT_ROOT/bin:$PATH

mapfile -t ALL_SOURCES < <(find "$SOURCE_DIR" -name "*.mseed" | sort -n )
SOURCE="${ALL_SOURCES[SLURM_ARRAY_TASK_ID]}"
SOURCE_NAME=$(basename "${SOURCE%.mseed}")

mapfile -t TEMPLATES_BATCHES < <(find "$TEMPLATES_BATCHES_DIR" -type d)
for TEMPLATES_BATCH in "${TEMPLATES_BATCHES[@]}"
do
  BATCH=$(basename "$TEMPLATES_BATCH")
  mkdir -p outputs
  OUTPUT=outputs/${SOURCE_NAME}_$BATCH
  export CUDA_VISIBLE_DEVICES=$((BATCH % 4)),
  correlation-detector "$SOURCE" "$TEMPLATES_BATCH/travel_times" "$TEMPLATES_BATCH/data" "$OUTPUT" &
done
wait

cat-detections outputs "$SOURCE_NAME"
mkdir -p "$OUTPUT_DIR"
cp "${SOURCE_NAME}.avro" "$OUTPUT_DIR"

exit
