# XRM data generator

`xrm_generator.py` makes **fake firmware amplitudes that look like real ones**:
42 numbers per bunch (one per channel) plus the true beam size σ_y and
position μ. The fake bunches are meant for training the neural network
that will run on the FPGA, and later as pseudo-data for testing it.

## Words used here

| Word | Meaning |
|---|---|
| **Sweep** | One full-turn capture (one ROOT file), e.g. `2025-12-08.180721` |
| **Template** | The *ideal* amplitudes a perfect detector would show for a given beam size and position, taken from the template library (`beam_profiles/smeared_grid_CMOS_upto200um.npz`) |
| **Reference** | The *best real measurement* of a bunch: amplitudes from the careful offline processing in `fit_from_rootfile.py` (interpolation, channel alignment, local baseline, gain) |
| **Firmware** | The *same real bunch* measured the cheaper way the FPGA will do it (`firmware_model.py`) |
| **Fit** | The template that best matches a bunch's reference amplitudes; gives σ_y, μ, brightness (`norm`), a constant offset (`v_offset`) and the image's placement on the sensor (`x0`, `scale`) |
| **Donor** | A real bunch whose fit parameters (all but σ_y) a fake bunch borrows |

## How to run it

```bash
python xrm_generator.py calibrate          # measure everything from real data -> data/generator_calibration.npz
python xrm_generator.py validate           # compare fakes of real bunches with sweeps not used for calibration
python xrm_generator.py generate -n 100000 --sigma-min 10 --sigma-max 200 -o train.npz
```

Two optional extras make the fake bunches more realistic; both are off by
default (the benchmark). `generate`, `validate`, `train_nn.py` and
`explain_one_bunch.py` all take them:

| Switch | What it adds | Why (see step) |
|---|---|---|
| `--factors` | multiplies the template by a correction factor per channel | real channels read consistently −33% to +26% off the template (5) |
| `--deviations` | adds the donor's real deviation: its real amplitudes minus what the generator makes for its own parameters | real bunches vary ~11% per channel from bunch to bunch (6) |

```bash
python xrm_generator.py generate -n 100000 --factors --deviations -o train_realistic.npz
python train_nn.py --rosters centre_12_27 --factors --deviations --output-dir data/nn_factors_deviations
```

`train_nn.py` records the switches next to each network, and
`plot_true_vs_predicted.py --model-dir ...` uses them to rebuild the
matching generator.

The output has the same keys as the old generator (`x_data`, `true_mu`,
`true_sig_y`), so `train_simple_nn.py` can read it. It also has
`in_real_sigma_range` (True where the bunch's beam size lies inside the
range covered by real data; see Limits) and `real_sigma_range_um`. `x_data` holds raw
firmware amplitudes for all 42 channels; choosing 16 channels and dividing
by their total happens at training time.

Calibration needs `data/amplitudes/`, made by `build_amplitude_dataset.py`,
which itself needs `data/real/` from `build_real_dataset.py`.

## How a fake bunch is made

```
 1. Parameters   pick a random real bunch (the donor)
                 └─ borrow its position, brightness, offset and image placement;
                    replace its σ_y with the requested one
 2. Template     ideal profile for those parameters
                 └─ --factors: × a correction factor per channel
                 └─ --deviations: + the donor's real deviation
 3. Firmware     × ~0.93 per channel + an offset + noise (1 ADC + 2%)
```

Steps 2–3 are the chain from ideal to what the FPGA sees:

```
 Template ── stands in for Reference ──(firmware response)──► Firmware
 (ideal)                   (best real)                         (what the FPGA sees)
```

`python explain_one_bunch.py --sigma 120` (add `--factors` / `--deviations`
as needed) walks through these steps for a single bunch, printing every intermediate number per channel and drawing
each step (`data/explain_one_bunch.png`). It checks at the end that the
result is identical to what the generator makes.

## How it was built, step by step

### Step 1: collect real data
`build_real_dataset.py` read all 680 full sweeps (Nov 27 – Dec 15, 2025).
584 are good, 40 have few bunches (kept apart), 56 failed or were
corrupted (excluded). For each bunch it stored the raw waveform values and
the offline fit.

### Step 2: decide how the firmware measures an amplitude
An amplitude is "the level the pulse sits on" minus "the pulse's lowest
point". The level the pulse sits on (the **baseline**) is the hard part:

- With no beam, each channel reads a fixed **pedestal** (10–17 ADC, from
  beam-off sweeps) plus random **noise** (~4.7 ADC RMS).
- With beam, the whole board shifts by a small **offset** per sweep (seen
  in the edge channels, which get no X-rays).
- Channels that receive X-rays are also pushed up during a bunch train, by
  an amount that grows with their signal (likely AC coupling in the
  amplifiers).

We tested baseline methods by how precisely the beam size could still be
fitted (`test_shift_bias.py`, `test_baseline_methods.py`). A fixed pedestal
made σ_y scatter 8.5 µm from bunch to bunch, against 4.9 µm for the offline
processing. The chosen method, **E50**, measures the level just before and
after each pulse and keeps a running average over ~50 bunches. It matches
the offline precision (4.9 µm) and is cheap in firmware. `firmware_model.py`
is the Python version of it.

### Step 3: three versions of every real bunch
`build_amplitude_dataset.py` stored, for every bunch and channel, the
**template**, **reference** and **firmware** amplitudes. Comparing them
tells us what the generator has to reproduce.

### Step 4: firmware vs reference (simple)
The firmware gives about **0.93 × the reference** in each channel (it
misses a little of each pulse's peak by not interpolating), plus a small
scatter of ~1 ADC + 2% of the amplitude. Part of that scatter is shared
by all channels. The generator applies the gain and offset (step 3
above) plus simple independent noise of 1 ADC + 2% per channel (no
shared part). Networks trained with no noise at all were much worse on
real data (per-bunch spread vs fit 7.5 and 10.7 µm for the two 16-channel
sets instead of 4.2 and 5.5; logs `data/nn/train_no_scatter_seed*.log`):
without noise they learn details real bunches don't have. With the simple
noise (3 seeds each, `data/nn/train_simple_noise_seed*.log`) they are back
to 4.3 and 5.7 µm, the same as with the full measured scatter model.

### Step 5: reference vs template (the hard part)
Real bunches differ from their best-fit template:

- **Consistently, per channel:** some channels always read higher or lower
  than the template, by −33% to +26%.
- **From bunch to bunch:** about 11%, with neighbouring channels moving
  somewhat together.

The consistent part changed during the first days (Nov 28 – Dec 3; the
setup seems to have been adjusted) and was steady from **Dec 4 to Dec 15**.
So only Dec 4–15 is used, split by alternating sweeps into a **calibration
half** and a **held-out half** for testing.

14 sweeps of that period are excluded (`EXCLUDED_SWEEPS` in the script):
Dec 4 ~15:05 and 16:18, Dec 5 ~21:15, Dec 8 ~12:20 and ~18:03. They have
normal current per bunch but 3–10× less detector signal and a shifted
image (μ ≈ 65–75 µm), and they sit right next to "beam on, no signal"
sweeps, so the detector or beamline was probably being changed. Excluding
them changed the correction factors by at most 0.25%. That leaves 219
calibration and 219 held-out sweeps.

From the calibration half, `measure_channel_factors.py` measured one **correction factor per channel**
(0.67–1.26). On held-out sweeps they remove the consistent mismatch almost
completely (15% → 0.3%). Whether these come from wrong gain constants or an
imperfect template shape couldn't be told apart, but for one setup they act
the same way.

**The factors are off by default (2026-10-07)**, to keep the generator
simple; `--factors` switches them on.
Networks trained without them (`archive/study_no_factors.py`, 3 seeds, log
`data/no_factors.log`) are worse on real held-out sweeps: per-bunch spread
vs fit 6.3 instead of 4.3 µm for centre 12–27, judged acceptable for now;
for every other 6–36 it is 10.6 instead of 5.6 µm and those networks read
beams ~12 µm too large, so **without factors only the centre channels are
usable**.

### Step 6: bunch-to-bunch variation (off by default, `--deviations`)
Real bunches also differ from their own corrected template by about 11%
per channel, differently from bunch to bunch. An earlier version copied
that difference from a real donor bunch with a similar beam size onto each
fake bunch (step 3 of 4). It was dropped on 2026-10-07 to keep the
generator simple enough to verify by reading, and came back as the
`--deviations` switch. Two small differences from that version: donors are
drawn at random instead of by matching σ_y (the donor still supplies its
own position, brightness and placement, so its deviation fits those), and
the scatter is the simple 1 ADC + 2% noise. The original version and the
comparison below stay in `archive/` for the record.

`archive/compare_deviation_step.py` trained the same network on both
versions (3 seeds each, identical random draws) and tested it on 110 real
held-out sweeps:

| Real held-out data | centre 12–27: with / without | every other 6–36: with / without |
|---|---|---|
| Network − fit, per-bunch spread | 2.41 / **4.20 µm** | 3.91 / **5.45 µm** |
| Scatter within a sweep (fit: 4.48) | 5.02 / 5.45 µm | 5.71 / 5.58 µm |
| Per sweep vs CMOS (fit: +1.29 ± 2.20) | +1.37 ± 2.59 / +2.20 ± 2.57 µm | +1.50 ± 2.79 / +1.93 ± **1.90** µm |

Without it, per-bunch precision is clearly worse, but per-sweep results
are about as good (for every other 6–36 the sweep-to-sweep agreement with
CMOS was even better; unexplained). Networks trained without it also fail
on generated data that has the variation: up to −23 µm off for large
beams. Logs: `data/deviation_comparison*.log`.

### Step 7: check against real data
`python xrm_generator.py validate` makes a fake of every real bunch in 60
held-out sweeps (from that bunch's own fit parameters) and compares
normalized shapes, as the network will see them:

| Check | Real | Generator | + factors (archived) | + factors + step 3 (archived) |
|---|---|---|---|---|
| Average shape, typical channel off by | — | 16.2% | 0.7% | 0.1% |
| Average shape, worst channel off by | — | 179% | 22.5% | 1.3% |
| Bunch-to-bunch spread per channel, fake ÷ real | 1 | 0.70 (0.38–2.99) | 0.69 (0.37–3.56) | 1.01 (0.88–1.45) |
| Neighbouring channels varying together | 0.19 | 0.56 | 0.56 | 0.20 |
| Fine structure (carries the σ_y information) | 0.368 | 0.176 | 0.310 | 0.367 |
| Its bunch-to-bunch spread | 0.050 | 0.026 | 0.023 | 0.050 |

The factors fix the average shape (step 5); step 3 adds the bunch-to-bunch
variation (step 6). The generator currently has neither.

## Benchmark (2026-10-07)

The reference result for this version of the generator: the centre
16-channel network (`centre_12_27`) trained only on generated data and
tested on real data it never saw. Any change to the generator should be
compared against these numbers in the same way.

**Generator recipe** (both switches off, the default): random real bunch's fit parameters with σ_y replaced
→ plain template (no channel factors) → per-channel firmware gain and
offset + independent noise of 1 ADC + 2%. Calibrated on the 219
calibration sweeps of Dec 4–15.

**Real test data:** every other held-out sweep (110 sweeps, 256,587
bunches), as in `train_nn.py`.

| Real held-out data | 3 seeds: mean (min..max) | Plotted network (seed 0) | Offline fit |
|---|---|---|---|
| Network − fit, median per bunch | +1.00 µm (+0.44..+1.48) | +1.06 µm | — |
| Network − fit, per-bunch spread | 6.25 µm (5.05..8.37) | 5.05 µm | — |
| Scatter within a sweep | 6.56 µm (5.41..8.39) | 5.41 µm | 4.48 µm |
| Per sweep vs CMOS | +2.24 ± 1.83 µm | +2.02 ± 1.54 µm | +1.29 ± 2.20 µm |

The 3-seed numbers are the benchmark: one network can be much better or
worse than another trained the same way. They come from
`archive/study_no_factors.py` ("no factors" column, `data/no_factors.log`),
whose recipe is exactly the current generator.

On generated data (true σ_y known), the plotted network's error spread is
1.3 µm inside the real range, 1.1–1.5 µm below it, 2.9 µm at 83–120 µm and
5.3 µm at 120–200 µm. Generated data only shows that the network learned
the generator; real-data performance outside 48–83 µm is unknown.

To reproduce:

```bash
python xrm_generator.py calibrate
python train_nn.py --seed 0 --rosters centre_12_27      # repeat with --seed 1 and 2
python plot_true_vs_predicted.py                        # data/nn/true_vs_predicted.png
```

## Limits

- **Default: no channel factors (2026-10-07):** real channels read consistently
  −33% to +26% off the template; by default the generator ignores this.
  Fine enough for the centre 16 channels, not for wider channel sets
  (step 5). `--factors` fixes it.
- **Default: no bunch-to-bunch variation (2026-10-07):** fake bunches are smooth
  templates plus small independent noise (1 ADC + 2%); `--deviations` adds it. Real bunches vary
  by ~11% per channel around their template, with neighbouring channels
  moving together. See step 6 for what this costs.
- **Beam size range (accepted for now, 2026-10-05):** the monitor should
  cover σ_y = 10–200 µm, and the generator does, but real bunches only
  exist for σ_y ≈ 48–83 µm, so only there can fake bunches be checked
  against real data. With a uniform 10–200 µm draw ~80% of bunches lie
  outside it. Each bunch is flagged (`in_real_sigma_range`), and
  training/evaluation results should be reported separately for the two
  ranges. More real data at small and large beam sizes would remove this
  limit.
- **One setup:** the gain/offset and donor parameters come from the Dec 4–15 setup. After any
  detector change (realignment, new gains), rerun the pipeline and
  `calibrate`.
- **Firmware is modelled, not measured on hardware.** The October board test
  will be the first real check of step 3.
- **Edge channels:** channels 0–3 and 40–41 get almost no X-rays;
  channels 35–39 are faint and less stable. They are
  likely poor choices among the 16 network inputs.

## Files

Pipeline, in order:

| File | Role |
|---|---|
| `build_real_dataset.py` | real sweeps → per-bunch raw values, offline fit, CMOS; sweep categories |
| `firmware_model.py` | Python model of the firmware amplitude (E50) |
| `build_amplitude_dataset.py` | template, reference and firmware amplitudes per bunch |
| `xrm_generator.py` | calibrate, generate, validate |
| `explain_one_bunch.py` | the generator's steps for one bunch, printed and drawn |
| `train_nn.py` | train networks on generated data, test on real held-out sweeps |
| `plot_true_vs_predicted.py` | true vs predicted σ_y for the centre network (benchmark plot) |
| `xrm_raw.py` | shared helpers (raw waveforms, beam-off pedestals, beam current) |

Studies that led to the design choices:

| File | Question it answered |
|---|---|
| `measure_sweep_offset.py` | How big is the board-wide offset, and how does it drift? |
| `measure_baseline_shift.py` | How does the baseline move along a bunch train? |
| `measure_sampling_loss.py` | How much of each peak is missed without interpolation? |
| `test_shift_bias.py` | Does a fixed pedestal bias the fitted σ_y? |
| `test_baseline_methods.py` | Which cheap baseline method keeps the precision? |
| `measure_generator_inputs.py` | Firmware-vs-reference response; first look at template mismatch |
| `measure_template_mismatch.py` | Is the template mismatch stable, and is it gain or shape? |
| `archive/compare_deviation_step.py` | Does the generator need real bunch-to-bunch variation? (needs `archive/xrm_generator_step3.py`) |
| `archive/study_no_factors.py` | What do the channel factors do for the networks? (needs `archive/xrm_generator_factors.py`) |
| `archive/study_baseline_donors.py` | Does a donor's averaged baseline explain its deviation? (~2%; no help to the network) |
| `archive/study_instant_baseline.py` | Does a donor's instantaneous baseline? (up to 61% on bright channels; network 3.70 vs 4.31 µm) |
| `measure_channel_factors.py` | Per-channel factors, tested on held-out sweeps and against CMOS |
