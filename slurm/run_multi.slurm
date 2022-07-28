#! /usr/bin/env bash
#SBATCH --job-name=<JOB_NAME>
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --time=<TIME>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<NUM_CPUS>
#SBATCH --gres=gpu:1
#SBATCH --output=<LOG_PATH>/%A_%a.log
#SBATCH	--requeue
#SBATCH --verbose

# NOTE: The Conda environment must be activated before submitting this script

DATA_ROOT=<DATA_ROOT>
DATA_DIR=$DATA_ROOT/<DATA>
TEMPLATES_DIR=$DATA_ROOT/<TEMPLATES>
TTIMES_DIR=$TEMPLATES_DIR/<TRAVEL_TIMES>
TEMPLATES_DATA_DIR=$TEMPLATES_DIR/<TEMPLATES_DATA>
CATALOGUE=$TEMPLATES_DIR/<CATALOGUE>

OUTPUT_DIR=$DATA_ROOT/<OUTPUT_DIR>
AVRO_DIR=$OUTPUT_ROOT/<AVRO>
PARQUET_DIR=$OUTPUT_ROOT/<PARQUET>
STATS_DIR=$OUTPUT_ROOT/<STATS>

DAYS_PER_TASK=<DAYS_PER_TASK>

mkdir -p "$AVRO_DIR"
mkdir -p "$PARQUET_DIR"
mkdir -p "$STATS_DIR"
mapfile -t DATA_BATCH < <(find "$DATA_DIR" -type f -name "*.mseed" | sort -n | head -n $((DAYS_PER_TASK * (SLURM_ARRAY_TASK_ID + 1))) | tail -n $DAYS_PER_TASK)
for DATA in "${DATA_BATCH[@]}"
do
  DATA_NAME=$(basename "${DATA%.mseed}")
  AVRO=$AVRO_DIR/$DATA_NAME
  PARQUET=$PARQUET_DIR/$DATA_NAME
  STATS=$STATS_DIR/$DATA_NAME
  srun eqtm-scan "$DATA" "$TTIMES_DIR" "$TEMPLATES_DATA_DIR" "$AVRO" && \
  eqtm-avro2parquet "$CATALOGUE" "${AVRO}.avro" "$PARQUET" && \
  eqtm-parquet2legacy "${PARQUET}.parquet" "$STATS" & || scontrol requeue $SLURM_JOB_ID
done
wait

exit
