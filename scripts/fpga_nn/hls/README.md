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
