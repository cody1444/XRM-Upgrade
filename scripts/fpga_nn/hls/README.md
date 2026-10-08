# Network → FPGA with hls4ml

Turns a network trained by `train_nn.py` into an HLS project for the
ZCU216 (xczu49dr-ffvf1760-2-e) with hls4ml.

## Steps

```bash
# 1. In xrm-tf: fold the input standardization and output scaling into the
#    weights, export them with a check set of real held-out bunches
PYTHONPATH=scripts/fpga_nn conda run -n xrm-tf python scripts/fpga_nn/hls/export_network.py \
    --model-dir scripts/fpga_nn/data/nn_deviations

# 2. In hls4ml (hls4ml 1.1, Keras 2): rebuild, convert, compare the C simulation with Keras
conda run -n hls4ml python scripts/fpga_nn/hls/convert_hls4ml.py

# 3. Vitis HLS C synthesis (resources, latency)
scripts/fpga_nn/hls/run_synthesis.sh scripts/fpga_nn/data/hls/centre_12_27_deviations_18bit
```

The two environments are needed because the hls4ml environment has
Keras 2, which can't read the Keras 3 `.keras` files; the network travels
as plain weights (`data/hls/<name>.npz`).

**Network input:** the 16 channel amplitudes (centre 12–27) divided by
their sum. That division is not part of the network: the firmware has to
do it, as training does. **Output:** σ_y and μ in µm.

Projects: `data/hls/<name>_wide/` (32-bit everywhere, a check of the
conversion) and `data/hls/<name>_18bit/` (the one for the FPGA). Both are
fully parallel (`io_parallel`, Latency strategy, reuse factor 1), clock
5 ns (200 MHz).

## The core: normalization + network (`hls/core/`)

`xrm_nn_core` is the IP core for the firmware. It takes the **16 raw
channel amplitudes** (ADC, channels 12–27), divides them by their sum and
runs the network:

```
raw[16] ─► sum ─► 1/sum (one divider) ─► raw[i] × 1/sum (16 DSPs) ─► network ─► σ_y, μ (µm)
```

If the sum is below 512 ADC (no beam), both outputs are 0.

| Port | Type | Meaning |
|---|---|---|
| `raw` (one 320-bit port, + valid) | 16 × `ap_fixed<20,14>` | amplitudes in ADC, ±8192, 1/64 ADC steps (real data: 41–3041) |
| `out_0`, `out_1` (+ valid) | `ap_fixed<18,9>` | σ_y and μ in µm |
| `ap_clk`, `ap_rst`, `ap_start`, `ap_done`, `ap_ready`, `ap_idle` | | clock, reset, HLS block handshake |

Pipelined: one bunch per clock. The input type is a guess until the
firmware's amplitude format is known; it is one `typedef` in
`xrm_nn_core.h` (`raw_t`), with the sum and reciprocal types next to it.

`convert_hls4ml.py` copies the core into the 18-bit project with 2000 real
held-out bunches as test vectors (`tb_data/core_input.dat`, raw amplitudes;
`tb_data/core_expected.dat`, Keras σ_y and μ), and checks it with g++. The
testbench `xrm_nn_core_test.cpp` fails if any output is off by more than
0.25 µm or if the no-beam input doesn't give 0.

**On the lab machine** (Vitis HLS 2023.1), after copying the regenerated
project folder over:

```bash
source /opt/Xilinx/Vitis_HLS/2023.1/settings64.sh && source /opt/Xilinx/Vivado/2023.1/settings64.sh
cd .../centre_12_27_deviations_18bit
vitis_hls -f build_core.tcl                       # csim, synthesis, co-simulation, IP export
```

| Step | Result |
|---|---|
| C simulation | `PASSED`/`FAILED` in the log; outputs in `tb_data/core_csim_results.log` |
| Synthesis | `xrm_nn_core_prj/solution1/syn/report/xrm_nn_core_csynth.rpt` |
| Co-simulation (the Verilog, same testbench) | `PASSED`/`FAILED` in the log; `xrm_nn_core_prj/solution1/sim/report/xrm_nn_core_cosim.rpt` |
| IP export | `xrm_nn_core_prj/solution1/impl/ip/` (`component.xml` + HDL): add this folder as an IP repository in Vivado (Settings → IP → Repository) |

Steps can be skipped, e.g. `vitis_hls -f build_core.tcl "cosim=0 export=0"`.

`build_core.tcl` keeps the network a separate block (`set_directive_inline
-off myproject`). Without that, Vitis inlines it into the core, the
per-layer multiplier limits hls4ml sets merge into the smallest one, and the
whole network shares 48 multipliers: a new bunch only every 65 clocks.

**Checked on the Virtex-7 stand-in** (Vitis HLS 2025.2, 2026-10-07): C
simulation and C/RTL co-simulation both PASS on 2000 real bunches (σ_y
0.028 µm rms, 0.062 µm max from Keras; μ 0.016 µm rms; no-beam → 0); one
bunch per clock; latency 77 cycles (network 32, the rest mostly the
divider); 2066 DSPs (network 2050 + normalization 16); IP export works.

**Built for the ZCU216** (xczu49dr-ffvf1760-2-e, Vitis HLS 2023.1 on the
lab machine, 2026-10-07): C simulation and C/RTL co-simulation PASS (same
2000 bunches and errors as above).

| | Core (normalization + network) | Network alone | Of the xczu49dr |
|---|---|---|---|
| Latency | 55 cycles = 275 ns | 14 cycles = 70 ns | |
| New bunch accepted every | 1 cycle (5 ns) | 1 cycle | |
| Estimated clock period | 3.64 ns (target 5 ns) | 3.39 ns | meets 200 MHz |
| DSP | 2,066 | 2,050 | 48% of 4,272 |
| LUT | 72,343 | 70,280 | 17% of 425,280 |
| FF | 39,255 | 36,058 | 4.6% of 850,560 |
| BRAM / URAM | 0 | 0 | |

(HLS estimates.) The normalization costs 16 DSPs and about 40 cycles of
latency, almost all of it the divider.

On that machine (Ubuntu with a newer glibc), Vivado 2023.1's own linker
can't read the system libraries (`unknown type [0x13] section .relr.dyn`),
so C simulation and co-simulation fail to link. Fixed there by pointing
`/opt/Xilinx/Vivado/2023.1/tps/lnx64/binutils-2.37/bin/ld` at `/usr/bin/ld`
(original kept as `ld.orig`). Synthesis and export don't need it.

## Fixed-point precision (18-bit version)

Chosen from the value ranges on real data, with about 2× headroom
(`PRECISION_18` in `convert_hls4ml.py`):

| Stage | Range on real data | Precision |
|---|---|---|
| Input (amplitude ÷ sum) | 0.005–0.16 | `ap_fixed<18,1>` |
| Layer 1 weights (standardization folded in) | ±86 | `ap_fixed<18,8>` |
| Layer 2–3 weights | ±1.6 | `ap_fixed<18,2>` |
| Layer 4 weights / bias (output scaling folded in) | ±34 / 107 | `ap_fixed<18,8>` |
| Hidden values | ±7.6 | `ap_fixed<18,5>` |
| Output (µm) | 38–136 (±256 allowed) | `ap_fixed<18,9>` |
| Sums inside a layer | — | result + 6 integer and 6 fractional bits |

## Results (2026-10-07, deviations-only centre network, seed 0)

C simulation vs Keras on 20,000 real held-out bunches:

| | σ_y error rms | σ_y error max | Network − fit, per-bunch spread |
|---|---|---|---|
| Keras | — | — | 2.007 µm |
| wide (`ap_fixed<32,12>`) | 0.0013 µm | 0.004 µm | 2.007 µm |
| 18-bit | 0.025 µm | 0.054 µm | 2.009 µm |

C synthesis, **on a Virtex-7 stand-in** (xc7vx690t-2; see below):

| | Value | On the ZCU216's xczu49dr |
|---|---|---|
| Latency | 32 cycles = 160 ns | |
| New bunch accepted every | 1 cycle (5 ns) | |
| Estimated clock period | 3.65 ns (target 5 ns) | UltraScale+ is faster |
| DSP | 2,050 | ~48% of 4,272 |
| LUT | 65,240 | ~15% of 425,280 |
| FF | 213,934 | ~25% of 850,560 |

C synthesis **for the ZCU216** (xczu49dr-ffvf1760-2-e, Vitis HLS 2023.1 on
the lab machine, same project):

| | Value | Of the xczu49dr |
|---|---|---|
| Latency | 14 cycles = 70 ns | |
| New bunch accepted every | 1 cycle (5 ns) | |
| Estimated clock period | 3.39 ns (target 5 ns, 1.35 ns uncertainty) | meets 200 MHz |
| DSP | 2,050 | 47% of 4,272 |
| LUT | 70,280 | 16% of 425,280 |
| FF | 36,058 | 4% of 850,560 |
| BRAM / URAM | 0 / 0 | |

Shorter latency and far fewer flip-flops than on the Virtex-7: UltraScale+
DSPs are faster, so fewer pipeline registers are needed.

Vivado 2023.1 synthesis of the same design (`vsynth=1`; utilization after
`synth_design` + `opt_design`, from `vivado_synth.rpt`):

| | HLS estimate | Vivado post-synthesis | Of the xczu49dr |
|---|---|---|---|
| DSP (DSP48E2) | 2,050 | 2,050 | 48% |
| LUT | 70,280 | 36,117 | 8.5% |
| FF | 36,058 | 26,590 | 3.1% |
| CARRY8 | — | 4,327 | 8.1% |
| BRAM / URAM / LUT as memory | 0 | 0 | 0% |

hls4ml's Vivado step reports utilization only, not timing: the 200 MHz
figure is still the HLS estimate (3.39 ns). A timing check needs Vivado
synthesis with a 5 ns clock constraint and a timing summary.

## Tool notes

- **ZCU216 device support is not installed** in `/opt/Xilinx/2025.2`
  (only 7-series parts are). Add Zynq UltraScale+ RFSoC with the AMD
  installer ("Add Design Tools or Devices"), then run step 3 on
  `centre_12_27_deviations_18bit`, which already targets the ZCU216 part.
  Until then, `data/hls/centre_12_27_deviations_18bit_virtex7/` is a copy
  with the part set to xc7vx690tffg1761-2.
- **Vitis 2025.2 has no `vitis_hls`**, which hls4ml 1.1 calls; its
  replacement `vitis-run --mode hls` can't pass arguments to a Tcl script.
  `run_synthesis.sh` works around both.
- `config_array_partition -maximum_size` no longer exists in Vitis HLS
  2025.2; hls4ml's script wraps it in `catch`, so the error in the log is
  harmless.
