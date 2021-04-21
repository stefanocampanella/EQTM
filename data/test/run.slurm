#! /usr/bin/env bash
#SBATCH --account IscrB_PALEOMIT
#SBATCH --partition m100_usr_prod
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=32GB
#SBATCH --hint=compute_bound
#SBATCH --mail-type=ALL
#SBATCH --verbose

export ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT:$PYTHONPATH
export PATH=$ROOT/bin:$PATH

cd "$ROOT/data/test" || exit
srun -n 1 correlation-detector data/2010-07-23.mseed ttimes templates templates.zmap test.stats "$@"

exit
