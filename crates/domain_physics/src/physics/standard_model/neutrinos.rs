//! Neutrino Physics: two-flavor, three-flavor PMNS, and constant-density MSW matter oscillations.

use super::flavor::construct_ckm;
use crate::error::StandardModelError;
use nalgebra::Matrix3;
use num_complex::Complex;

/// Conversion factor $\frac{10^3\ \text{m}}{2 \cdot 10^9\ \text{eV} \cdot \hbar c} = 2.533_865_358$
/// mapping $\Delta m^2\ [\text{eV}^2] \cdot L\ [\text{km}] / E\ [\text{GeV}]$ to dimensionless phase $2 \phi_{ij}$.
const OSC_PHASE_FACTOR: f64 = 2.533_865_358_139_463;

/// Wolfenstein matter potential prefactor $2 \sqrt{2} G_F N_A\ [\text{eV}^2 / (\text{GeV} \cdot \text{g/cm}^3)]$.
const MSW_POTENTIAL_FACTOR_EV2: f64 = 1.526_494_0e-4;

/// Weak-interaction neutrino flavor eigenstate ($\nu_e, \nu_\mu, \nu_\tau$).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeutrinoFlavor {
    /// Electron neutrino ($\nu_e$).
    Electron,
    /// Muon neutrino ($\nu_\mu$).
    Muon,
    /// Tau neutrino ($\nu_\tau$).
    Tau,
}

impl NeutrinoFlavor {
    /// Returns the zero-based flavor index (`0 = e`, `1 = mu`, `2 = tau`).
    #[verified_engine::verified]
    pub const fn index(self) -> usize {
        match self {
            Self::Electron => 0,
            Self::Muon => 1,
            Self::Tau => 2,
        }
    }
}

/// Three-flavor Pontecorvo-Maki-Nakagawa-Sakata (PMNS) oscillation parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PmnsParameters {
    /// Solar mixing angle $\theta_{12}$ (radians).
    pub theta12: f64,
    /// Atmospheric mixing angle $\theta_{23}$ (radians).
    pub theta23: f64,
    /// Reactor mixing angle $\theta_{13}$ (radians).
    pub theta13: f64,
    /// Dirac CP-violating phase $\delta_{\text{CP}}$ (radians).
    pub delta_cp: f64,
    /// Solar mass-squared splitting $\Delta m^2_{21} = m_2^2 - m_1^2$ ($\text{eV}^2$).
    pub delta_m21_sq: f64,
    /// Atmospheric mass-squared splitting $\Delta m^2_{31} = m_3^2 - m_1^2$ ($\text{eV}^2$).
    pub delta_m31_sq: f64,
}

impl Default for PmnsParameters {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::pdg_normal_ordering()
    }
}

impl PmnsParameters {
    /// Standard PDG / NuFIT normal-ordering best-fit oscillation parameters.
    #[verified_engine::verified]
    pub fn pdg_normal_ordering() -> Self {
        Self {
            theta12: 33.41_f64.to_radians(),
            theta23: 49.10_f64.to_radians(),
            theta13: 8.54_f64.to_radians(),
            delta_cp: 197.0_f64.to_radians(),
            delta_m21_sq: 7.41e-5,
            delta_m31_sq: 2.507e-3,
        }
    }

    /// Standard PDG / NuFIT inverted-ordering best-fit oscillation parameters.
    #[verified_engine::verified]
    pub fn pdg_inverted_ordering() -> Self {
        Self {
            delta_m31_sq: -2.486e-3,
            ..Self::pdg_normal_ordering()
        }
    }

    #[verified_engine::verified]
    fn validate(&self) -> Result<(), StandardModelError> {
        let all_finite = self.theta12.is_finite()
            && self.theta23.is_finite()
            && self.theta13.is_finite()
            && self.delta_cp.is_finite()
            && self.delta_m21_sq.is_finite()
            && self.delta_m31_sq.is_finite();
        if !all_finite {
            return Err(StandardModelError::InvalidNeutrinoParameter {
                reason: "PMNS angles and mass-squared splittings must be finite",
            });
        }
        Ok(())
    }
}

/// Constructs the $3 \times 3$ unitary PMNS lepton mixing matrix $U_{\text{PMNS}}$
/// by reusing the PDG unitary parameterization [`construct_ckm`].
#[verified_engine::verified]
pub fn construct_pmns(params: &PmnsParameters) -> Matrix3<Complex<f64>> {
    construct_ckm(
        params.theta12,
        params.theta23,
        params.theta13,
        params.delta_cp,
    )
}

/// Computes the Wolfenstein charged-current effective matter potential
/// $A_{CC} = \pm 2 \sqrt{2} G_F N_e E$ in $\text{eV}^2$.
#[verified_engine::verified]
pub fn matter_potential_ev2(
    e_gev: f64,
    density_g_cm3: f64,
    electron_fraction: f64,
    antineutrino: bool,
) -> Result<f64, StandardModelError> {
    if !e_gev.is_finite() || e_gev <= 0.0 {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Neutrino energy E (GeV) must be finite and strictly positive",
        });
    }
    if !density_g_cm3.is_finite() || density_g_cm3 < 0.0 {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Matter density (g/cm^3) must be finite and non-negative",
        });
    }
    if !electron_fraction.is_finite() || !(0.0..=1.0).contains(&electron_fraction) {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Electron fraction Y_e must lie in [0, 1]",
        });
    }
    let sign = if antineutrino { -1.0 } else { 1.0 };
    Ok(sign * MSW_POTENTIAL_FACTOR_EV2 * e_gev * density_g_cm3 * electron_fraction)
}

/// Computes three-flavor vacuum oscillation probabilities $[P_e, P_\mu, P_\tau]$ for an `initial`
/// neutrino (or antineutrino when `antineutrino = true`, which conjugates $U_{\text{PMNS}}$).
#[verified_engine::verified]
pub fn pmns_oscillation_probabilities(
    params: &PmnsParameters,
    initial: NeutrinoFlavor,
    l_km: f64,
    e_gev: f64,
    antineutrino: bool,
) -> Result<[f64; 3], StandardModelError> {
    params.validate()?;
    if !l_km.is_finite() || l_km < 0.0 {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Baseline distance L (km) must be finite and non-negative",
        });
    }
    if !e_gev.is_finite() || e_gev <= 0.0 {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Neutrino energy E (GeV) must be finite and strictly positive",
        });
    }
    let u = construct_pmns(params);
    let scale = OSC_PHASE_FACTOR * l_km / e_gev;
    let phases = [
        Complex::new(1.0, 0.0),
        Complex::from_polar(1.0, -scale * params.delta_m21_sq),
        Complex::from_polar(1.0, -scale * params.delta_m31_sq),
    ];

    let alpha = initial.index();
    let mut probs = [0.0; 3];
    let mut beta = 0;
    while beta < 3 {
        let mut amp = Complex::new(0.0, 0.0);
        let mut k = 0;
        while k < 3 {
            let u_bk = if antineutrino {
                u[(beta, k)].conj()
            } else {
                u[(beta, k)]
            };
            let u_ak_conj = if antineutrino {
                u[(alpha, k)]
            } else {
                u[(alpha, k)].conj()
            };
            amp += u_bk * phases[k] * u_ak_conj;
            k += 1;
        }
        probs[beta] = amp.norm_sqr();
        beta += 1;
    }
    Ok(probs)
}

/// Computes three-flavor constant-density MSW matter oscillation probabilities $[P_e, P_\mu, P_\tau]$
/// via stack-allocated scaling-and-squaring unitary evolution $S = \exp(-i H_{\text{MSW}} L / \hbar c)$.
#[verified_engine::verified]
pub fn msw_oscillation_probabilities(
    params: &PmnsParameters,
    initial: NeutrinoFlavor,
    l_km: f64,
    e_gev: f64,
    density_g_cm3: f64,
    electron_fraction: f64,
    antineutrino: bool,
) -> Result<[f64; 3], StandardModelError> {
    params.validate()?;
    if !l_km.is_finite() || l_km < 0.0 {
        return Err(StandardModelError::InvalidNeutrinoParameter {
            reason: "Baseline distance L (km) must be finite and non-negative",
        });
    }
    let a_cc = matter_potential_ev2(e_gev, density_g_cm3, electron_fraction, antineutrino)?;
    if density_g_cm3 == 0.0 || electron_fraction == 0.0 {
        return pmns_oscillation_probabilities(params, initial, l_km, e_gev, antineutrino);
    }

    let u_raw = construct_pmns(params);
    let u = if antineutrino {
        u_raw.map(|c| c.conj())
    } else {
        u_raw
    };
    let diag_m2 = Matrix3::from_diagonal(&nalgebra::Vector3::new(
        Complex::new(0.0, 0.0),
        Complex::new(params.delta_m21_sq, 0.0),
        Complex::new(params.delta_m31_sq, 0.0),
    ));
    let mut h_eff = u * diag_m2 * u.adjoint();
    h_eff[(0, 0)] += Complex::new(a_cc, 0.0);

    let phase_scale = OSC_PHASE_FACTOR * l_km / e_gev;
    let s_mat = unitary_exp_neg_i_hermitian(&(h_eff * Complex::new(phase_scale, 0.0)));
    let alpha = initial.index();
    Ok([
        s_mat[(0, alpha)].norm_sqr(),
        s_mat[(1, alpha)].norm_sqr(),
        s_mat[(2, alpha)].norm_sqr(),
    ])
}

/// Evaluates $\exp(-i K)$ for a $3 \times 3$ Hermitian generator $K$ on stack without heap allocation.
#[verified_engine::verified]
fn unitary_exp_neg_i_hermitian(k_mat: &Matrix3<Complex<f64>>) -> Matrix3<Complex<f64>> {
    let mean_trace = (k_mat[(0, 0)].re + k_mat[(1, 1)].re + k_mat[(2, 2)].re) / 3.0;
    let id = Matrix3::<Complex<f64>>::identity();
    let shifted = *k_mat - id * Complex::new(mean_trace, 0.0);

    let mut norm1 = 0.0_f64;
    let mut col = 0;
    while col < 3 {
        let col_sum =
            shifted[(0, col)].norm() + shifted[(1, col)].norm() + shifted[(2, col)].norm();
        if col_sum > norm1 {
            norm1 = col_sum;
        }
        col += 1;
    }

    let mut squarings = 0u32;
    let mut scaled_norm = norm1;
    while scaled_norm > 0.25 && squarings < 52 {
        scaled_norm *= 0.5;
        squarings += 1;
    }
    let scale = 0.5_f64.powi(squarings as i32);
    let m_mat = shifted * Complex::new(0.0, -scale);

    // 14-term Horner Taylor series for exp(M): |M|^15 / 15! < 7e-22 for |M| <= 0.25.
    let mut exp_m = id;
    let mut term_deg = 14usize;
    while term_deg >= 1 {
        exp_m = id + (m_mat * exp_m) * Complex::new(1.0 / (term_deg as f64), 0.0);
        term_deg -= 1;
    }

    let mut sq = 0;
    while sq < squarings {
        exp_m *= exp_m;
        sq += 1;
    }
    exp_m
}

/// Calculates the probability of two-flavor neutrino oscillation $P(\nu_\alpha \to \nu_\beta)$.
///
/// # Arguments
/// * `theta`: The mixing angle $\theta$ (radians).
/// * `delta_m2`: The mass-squared difference $\Delta m^2$ (eV^2).
/// * `l_km`: The baseline distance $L$ (km).
/// * `e_gev`: The neutrino energy $E$ (GeV).
///
/// # Formula
/// $P = \sin^2(2\theta) \sin^2\left( 1.27 \frac{\Delta m^2 L}{E} \right)$
#[verified_engine::verified]
pub fn oscillation_prob(theta: f64, delta_m2: f64, l_km: f64, e_gev: f64) -> f64 {
    let term1 = (2.0 * theta).sin().powi(2);
    // The factor 1.27 comes from conversion of units: 1.27 * (L/km) * (dm^2/eV^2) / (E/GeV)
    let phase = 1.27 * delta_m2 * l_km / e_gev;
    let term2 = phase.sin().powi(2);
    term1 * term2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_neutrino_oscillation() {
        let p = oscillation_prob(0.5, 0.0025, 295.0, 0.6);
        assert!((0.0..=1.0).contains(&p));
        assert_relative_eq!(oscillation_prob(0.0, 0.0025, 295.0, 0.6), 0.0);
        assert_relative_eq!(oscillation_prob(0.5, 0.0, 295.0, 0.6), 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_pmns_and_msw_unitarity_and_limits() {
        let params = PmnsParameters::pdg_normal_ordering();
        let u = construct_pmns(&params);
        let prod = u * u.adjoint();
        assert_relative_eq!(prod[(0, 0)].re, 1.0, epsilon = 1e-12);
        assert_relative_eq!(prod[(0, 1)].norm(), 0.0, epsilon = 1e-12);

        // Vacuum 3-flavor vs MSW at zero density match to machine precision and sum to 1.
        let vac = pmns_oscillation_probabilities(&params, NeutrinoFlavor::Muon, 1300.0, 2.5, false)
            .unwrap();
        let msw_zero = msw_oscillation_probabilities(
            &params,
            NeutrinoFlavor::Muon,
            1300.0,
            2.5,
            0.0,
            0.5,
            false,
        )
        .unwrap();
        let msw_crust = msw_oscillation_probabilities(
            &params,
            NeutrinoFlavor::Muon,
            1300.0,
            2.5,
            2.8,
            0.5,
            false,
        )
        .unwrap();
        let msw_anti = msw_oscillation_probabilities(
            &params,
            NeutrinoFlavor::Muon,
            1300.0,
            2.5,
            2.8,
            0.5,
            true,
        )
        .unwrap();

        assert_relative_eq!(vac[0] + vac[1] + vac[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(
            msw_crust[0] + msw_crust[1] + msw_crust[2],
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            msw_anti[0] + msw_anti[1] + msw_anti[2],
            1.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(vac[0], msw_zero[0], epsilon = 1e-12);

        // In normal ordering for DUNE-like L=1300 km, E=2.5 GeV, crust matter enhances P(nu_mu -> nu_e) over antineutrinos!
        assert!(msw_crust[0] > vac[0]);
        assert!(msw_crust[0] > msw_anti[0]);
    }
}
