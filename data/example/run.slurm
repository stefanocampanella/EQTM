#! /usr/bin/env bash
#SBATCH --account=IscrB_PALEOMIT
#SBATCH --partition=m100_usr_preempt 
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1800MB
#SBATCH --hint=compute_bound
#SBATCH --mail-type=ALL
#SBATCH --verbose

module load git

export ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT:$PYTHONPATH
export PATH=$ROOT/bin:$PATH

cd "$ROOT/data/example" || exit
for datafile in data/*.mseed
do
  bname=$(basename $datafile)
  correlation-detector $datafile templates/travel_times templates/data templates/catalog.zmap ${bname%.mseed} --log=info &
done
wait

exit
