#ifndef XRM_NN_CORE_H_
#define XRM_NN_CORE_H_

// XRM beam-size core: 16 raw channel amplitudes in, sigma_y and mu (um) out.
//
//   1. sum the 16 amplitudes
//   2. one reciprocal 1/sum
//   3. normalized[i] = raw[i] * (1/sum)    (as in training: amplitudes / their sum)
//   4. the hls4ml network (myproject)
//
// If the sum is below MIN_SUM_ADC (no beam), the outputs are 0.

#include "ap_fixed.h"
#include "myproject.h"

#define XRM_N_CHANNELS 16
#define XRM_N_OUTPUTS 2

// Raw amplitude per channel, in ADC. Real data on channels 12-27: 41-3041 ADC.
typedef ap_fixed<20, 14> raw_t;      // +-8192 ADC, 1/64 ADC resolution
// Sum of the 16 amplitudes. Real data: 5115-22449 ADC.
typedef ap_fixed<24, 18> sum_t;
// 1/sum. Covers sums down to 512 ADC (2^9).
typedef ap_ufixed<18, -9> recip_t;
// Numerator for the division: enough fractional bits for an accurate 1/sum.
typedef ap_ufixed<40, 1> one_t;

const int MIN_SUM_ADC = 512;

void xrm_nn_core(raw_t raw[XRM_N_CHANNELS], result_t out[XRM_N_OUTPUTS]);

#endif
