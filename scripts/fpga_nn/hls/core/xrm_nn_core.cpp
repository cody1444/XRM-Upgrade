#include "xrm_nn_core.h"

void xrm_nn_core(raw_t raw[XRM_N_CHANNELS], result_t out[XRM_N_OUTPUTS]) {
    // Same interface as the hls4ml network: all 16 inputs in one wide port,
    // each output its own port, valid signals, one new bunch per clock.
    #pragma HLS ARRAY_RESHAPE variable=raw complete dim=0
    #pragma HLS ARRAY_PARTITION variable=out complete dim=0
    #pragma HLS INTERFACE ap_vld port=raw,out
    #pragma HLS PIPELINE II=1

    // 1. Sum
    sum_t sum = 0;
    for (int i = 0; i < XRM_N_CHANNELS; i++) {
        #pragma HLS UNROLL
        sum += raw[i];
    }

    // No beam: no meaningful profile.
    if (sum < MIN_SUM_ADC) {
        for (int j = 0; j < XRM_N_OUTPUTS; j++) {
            #pragma HLS UNROLL
            out[j] = 0;
        }
        return;
    }

    // 2. One reciprocal instead of 16 divisions
    one_t one = 1;
    recip_t recip = one / sum;

    // 3. Normalize
    input_t normalized[XRM_N_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=normalized complete dim=0
    for (int i = 0; i < XRM_N_CHANNELS; i++) {
        #pragma HLS UNROLL
        normalized[i] = raw[i] * recip;
    }

    // 4. Network
    myproject(normalized, out);
}
