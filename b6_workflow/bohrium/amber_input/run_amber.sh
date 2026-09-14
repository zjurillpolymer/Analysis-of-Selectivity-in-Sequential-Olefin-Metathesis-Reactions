#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
exec > >(tee -a amber_job.log) 2>&1

status=0
archive_results() {
  status=$?
  tar -czf B6_neighbor_amber_results.tar.gz workspace
  exit "$status"
}
trap archive_results EXIT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

command -v antechamber
command -v sqm
command -v parmchk2
python3 workspace/analysis/setup_explicit_dcm_gromacs.py \
  --root workspace \
  --states \
    B6_p_C6_q1_V1_V2_cis \
    B6_p_C6_q2_neighbor_full_reuse_cis \
    B6_p_C6_q2_neighbor_one_arm_reuse_cis \
    B6_p_C6_q2_neighbor_full_switch_d1_cis \
    B6_p_C6_q2_neighbor_full_switch_d2_cis \
  --jobs 1 \
  --dcm-model ch2cl2_dc \
  --sqm-grms-tol 0.05 \
  --parameterize-only
