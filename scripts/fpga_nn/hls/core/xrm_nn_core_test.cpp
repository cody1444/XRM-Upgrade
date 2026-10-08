// Self-checking testbench for xrm_nn_core.
//
// tb_data/core_input.dat     one real held-out bunch per line: 16 raw amplitudes (ADC)
// tb_data/core_expected.dat  the original Keras network's sigma_y and mu (um) for it
//
// Runs every bunch through the core and compares. Fails (exit code 1) if any
// sigma_y or mu is off by more than TOLERANCE_UM, or if a no-beam input
// (all amplitudes 0) doesn't give 0. Used for C simulation and, with RTL_SIM
// defined, for C/RTL co-simulation.

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "firmware/xrm_nn_core.h"

// hls4ml's layer tracing (declared in nnet_helpers.h), off.
namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

const double TOLERANCE_UM = 0.25;

int main() {
#ifdef RTL_SIM
    const char *log_name = "tb_data/core_rtl_cosim_results.log";
#else
    const char *log_name = "tb_data/core_csim_results.log";
#endif
    std::ifstream fin("tb_data/core_input.dat");
    std::ifstream fexp("tb_data/core_expected.dat");
    std::ofstream flog(log_name);
    if (!fin.is_open() || !fexp.is_open()) {
        std::cerr << "ERROR: missing tb_data/core_input.dat or tb_data/core_expected.dat" << std::endl;
        return 1;
    }

    int n = 0, failed = 0;
    double max_err[XRM_N_OUTPUTS] = {0, 0}, sum_sq[XRM_N_OUTPUTS] = {0, 0};
    std::string in_line, exp_line;
    while (std::getline(fin, in_line) && std::getline(fexp, exp_line)) {
        std::istringstream in_s(in_line), exp_s(exp_line);
        raw_t raw[XRM_N_CHANNELS];
        for (int i = 0; i < XRM_N_CHANNELS; i++) {
            double v;
            in_s >> v;
            raw[i] = v;
        }
        double expected[XRM_N_OUTPUTS];
        for (int j = 0; j < XRM_N_OUTPUTS; j++)
            exp_s >> expected[j];

        result_t out[XRM_N_OUTPUTS];
        xrm_nn_core(raw, out);

        for (int j = 0; j < XRM_N_OUTPUTS; j++) {
            double err = out[j].to_double() - expected[j];
            max_err[j] = std::fmax(max_err[j], std::fabs(err));
            sum_sq[j] += err * err;
            if (std::fabs(err) > TOLERANCE_UM)
                failed++;
            flog << out[j].to_double() << (j + 1 < XRM_N_OUTPUTS ? " " : "\n");
        }
        n++;
    }

    // No beam: all amplitudes 0 -> outputs 0.
    raw_t zeros[XRM_N_CHANNELS];
    for (int i = 0; i < XRM_N_CHANNELS; i++)
        zeros[i] = 0;
    result_t out0[XRM_N_OUTPUTS];
    xrm_nn_core(zeros, out0);
    bool zero_ok = (out0[0] == 0) && (out0[1] == 0);

    std::cout << "xrm_nn_core: " << n << " real bunches vs Keras" << std::endl;
    const char *names[XRM_N_OUTPUTS] = {"sigma_y", "mu"};
    for (int j = 0; j < XRM_N_OUTPUTS; j++)
        std::cout << "  " << names[j] << ": rms error " << std::sqrt(sum_sq[j] / n) << " um, max " << max_err[j]
                  << " um" << std::endl;
    std::cout << "  no-beam input gives 0: " << (zero_ok ? "yes" : "NO") << std::endl;

    if (n == 0 || failed > 0 || !zero_ok) {
        std::cout << "FAILED (" << failed << " outputs off by more than " << TOLERANCE_UM << " um)" << std::endl;
        return 1;
    }
    std::cout << "PASSED (all outputs within " << TOLERANCE_UM << " um)" << std::endl;
    return 0;
}
