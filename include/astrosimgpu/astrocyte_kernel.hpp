#pragma once

#include <cmath>

#include "astrosimgpu/rng.hpp"
#include "astrosimgpu/types.hpp"

namespace astrosimgpu {

/// Per-cell astrocyte update, free of member state.
///
/// The functions here take plain scalars and nothing else: no `this`, no
/// `std::vector`, no reference into a class. That is what makes them callable
/// from a device kernel, and it is the whole reason they were lifted out of
/// AstrocytePopulation.
///
/// The host loop and any offloaded loop call the *same* function, so the two
/// cannot drift apart. The maths is covered by the existing tests through the
/// host path; only the mapping directives are new when offload is enabled.

/// Parameters shared by every astrocyte, small enough to pass by value into a
/// kernel rather than mapped as a pointer.
struct AstroConstants {
    real Kd_IP3_1;
    real Kd_IP3_2;
    real Kd_act;
    real Kd_inh;
    real Km_SERCA;
    real k_IP3R;
    real rate_IP3R;
    real rate_L;
    real rate_SERCA;
    real ratio_ER_cyt;
};

#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp declare target
#endif

/// Right-hand side of the three-variable Li-Rinzel system for one cell.
inline void astro_derivatives(const AstroConstants& c, real Ca_tot, real IP3_0, real tau_IP3,
                              real Ca, real IP3, real h, real noise, real& dCa, real& dIP3,
                              real& dh) {
    const real Ca_ER = (Ca_tot - Ca) / c.ratio_ER_cyt;

    const real m_inf = IP3 / (IP3 + c.Kd_IP3_1);
    const real n_inf = Ca / (Ca + c.Kd_act);
    const real m3 = m_inf * m_inf * m_inf;
    const real n3 = n_inf * n_inf * n_inf;
    const real h3 = h * h * h;

    const real J_channel = c.ratio_ER_cyt * c.rate_IP3R * m3 * n3 * h3 * (Ca_ER - Ca);
    const real J_pump = c.rate_SERCA * Ca * Ca / (c.Km_SERCA * c.Km_SERCA + Ca * Ca);
    const real J_leak = c.ratio_ER_cyt * c.rate_L * (Ca_ER - Ca);

    const real alpha_h = c.k_IP3R * c.Kd_inh * (IP3 + c.Kd_IP3_1) / (IP3 + c.Kd_IP3_2);
    const real beta_h = c.k_IP3R * Ca;

    dCa = J_channel - J_pump + J_leak + noise;
    dIP3 = (IP3_0 - IP3) / tau_IP3;
    dh = alpha_h * (1.0 - h) - beta_h * h;
}

/// Advance one astrocyte across a whole communication step.
///
/// State is taken and returned by reference so it can live in registers for
/// the duration: nothing here reads or writes memory between substeps, which
/// is the property the offloaded version depends on.
inline void astro_advance(const AstroConstants& c, real Ca_tot, real IP3_0, real tau_IP3,
                          real delta_IP3, real ip3_input, real noise, real h_step, int substeps,
                          real& Ca, real& IP3, real& h) {
    // Spikes arriving this step deposit IP3 instantaneously; the synaptic
    // drive enters the IP3 equation as a sum of delta functions.
    if (ip3_input != 0.0) {
        IP3 += delta_IP3 * ip3_input;
    }

    for (int s = 0; s < substeps; ++s) {
        real k1Ca, k1IP3, k1h, k2Ca, k2IP3, k2h, k3Ca, k3IP3, k3h, k4Ca, k4IP3, k4h;

        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca, IP3, h, noise, k1Ca, k1IP3, k1h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + 0.5 * h_step * k1Ca,
                          IP3 + 0.5 * h_step * k1IP3, h + 0.5 * h_step * k1h, noise, k2Ca, k2IP3,
                          k2h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + 0.5 * h_step * k2Ca,
                          IP3 + 0.5 * h_step * k2IP3, h + 0.5 * h_step * k2h, noise, k3Ca, k3IP3,
                          k3h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + h_step * k3Ca, IP3 + h_step * k3IP3,
                          h + h_step * k3h, noise, k4Ca, k4IP3, k4h);

        Ca += (h_step / 6.0) * (k1Ca + 2.0 * k2Ca + 2.0 * k3Ca + k4Ca);
        IP3 += (h_step / 6.0) * (k1IP3 + 2.0 * k2IP3 + 2.0 * k3IP3 + k4IP3);
        h += (h_step / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h);

        // Calcium is conserved between cytosol and ER, so the cytosolic
        // concentration is confined to [0, Ca_tot]; h is a fraction.
        Ca = Ca < 0.0 ? 0.0 : (Ca > Ca_tot ? Ca_tot : Ca);
        h = h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
        IP3 = IP3 < 0.0 ? 0.0 : IP3;
    }
}

#ifdef ASTROSIMGPU_OFFLOAD
#pragma omp end declare target
#endif

}  // namespace astrosimgpu
