#! /usr/bin/env bash
#SBATCH --account IscrB_PALEOMIT
#SBATCH --partition m100_usr_prod
#SBATCH --qos=m100_qos_dbg
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1GB
#SBATCH --hint=compute_bound
#SBATCH --mail-type=ALL
#SBATCH --verbose

module load git

export ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$ROOT:$PYTHONPATH
export PATH=$ROOT/bin:$PATH

cp -r  "$ROOT/data/test" "$TMPDIR"
cd "$TMPDIR" || exit
srun -n 1 correlation-detector data/2010-07-23.mseed ttimes templates templates.zmap "$@"
cp "$TMPDIR/$1" "$ROOT/$1"

exit
