# Build the XRM beam-size core: xrm_nn_core = normalization + the hls4ml network.
# Run from the hls4ml project folder (convert_hls4ml.py copies this file there):
#
#   vitis_hls -f build_core.tcl                         all steps
#   vitis_hls -f build_core.tcl "cosim=0 export=0"      C simulation and synthesis only
#
# Steps: csim   C simulation, self-checking against Keras (tb_data/core_*.dat)
#        synth  C synthesis -> xrm_nn_core_prj/solution1/syn/report/xrm_nn_core_csynth.rpt
#        cosim  C/RTL co-simulation: the same testbench on the generated Verilog
#        export Vivado IP catalog package -> xrm_nn_core_prj/solution1/impl/ip/

array set opt {csim 1 synth 1 cosim 1 export 1}
foreach arg [split [join $::argv " "]] {
    if {[regexp {^(\w+)=(\w+)$} $arg -> key value] && [info exists opt($key)]} {
        set opt($key) $value
    }
}
puts "build_core.tcl: csim=$opt(csim) synth=$opt(synth) cosim=$opt(cosim) export=$opt(export)"

# part, clock_period, clock_uncertainty from the hls4ml project
source [file join [file dirname [info script]] project.tcl]

open_project -reset xrm_nn_core_prj
set_top xrm_nn_core
add_files firmware/xrm_nn_core.cpp -cflags "-std=c++0x"
add_files firmware/myproject.cpp -cflags "-std=c++0x"
add_files -tb xrm_nn_core_test.cpp -cflags "-std=c++0x"
add_files -tb firmware/weights
add_files -tb tb_data
open_solution -reset solution1
set_part $part
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default
config_compile -name_max_length 80
# Keep the hls4ml network a block of its own. Inlined into xrm_nn_core, its
# per-layer multiplier limits merge into one (the smallest) and the network
# shares 48 multipliers: a new bunch only every 65 clocks instead of every clock.
set_directive_inline -off myproject

if {$opt(csim)} {
    puts "***** C SIMULATION *****"
    csim_design
}
if {$opt(synth)} {
    puts "***** C SYNTHESIS *****"
    csynth_design
}
if {$opt(cosim)} {
    puts "***** C/RTL CO-SIMULATION *****"
    add_files -tb xrm_nn_core_test.cpp -cflags "-std=c++0x -DRTL_SIM"
    cosim_design
}
if {$opt(export)} {
    puts "***** EXPORT IP *****"
    export_design -format ip_catalog -vendor "xrm" -library xrm -version 1.0 \
        -description "XRM beam size (sigma_y, mu) from 16 channel amplitudes"
}
exit
