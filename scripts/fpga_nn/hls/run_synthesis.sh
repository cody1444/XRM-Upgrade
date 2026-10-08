#!/usr/bin/env bash
# Run Vitis HLS C synthesis on an hls4ml project (from convert_hls4ml.py).
#
#   scripts/fpga_nn/hls/run_synthesis.sh scripts/fpga_nn/data/hls/centre_12_27_deviations_18bit
#
# Vitis 2025.2 has no vitis_hls command, which hls4ml 1.1 expects; its
# replacement, vitis-run, can't pass arguments to a Tcl script. So this
# writes a small wrapper that sets the arguments build_prj.tcl reads from
# argv, and runs that. Options can be overridden, e.g. "cosim=1".

set -euo pipefail
project_dir=${1:?usage: run_synthesis.sh <hls4ml project dir> [option=value ...]}
shift
# (Don't pass vsynth=...: build_prj.tcl matches option names anywhere in an
# argument, so "vsynth=0" would also set synth=0.)
options="reset=1 csim=0 synth=1 cosim=0 validation=0 export=0 $*"

set +u   # the Xilinx settings script uses unset variables
source /opt/Xilinx/2025.2/Vitis/settings64.sh
set -u
cd "$project_dir"
cat > run_hls.tcl <<EOF
set ::argv {$options}
source build_prj.tcl
EOF
vitis-run --mode hls --tcl run_hls.tcl
