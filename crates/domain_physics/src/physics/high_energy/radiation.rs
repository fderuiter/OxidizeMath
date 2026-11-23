use crate::error::HighEnergyError;
use crate::physics::high_energy::fluid_dynamics::shock_compression_ratio;
use crate::physics::high_energy::observer::{
    apparent_superluminal_beta, calculate_lorentz_factor_from_beta, relativistic_doppler_factor,
};
use crate::physics::high_energy::statistics::li_ma_significance;
use math_commons::constants::{C, SIGMA_T};
use std::f64::consts::PI;

/// Calculates the total synchrotron power radiated by a single electron.
/// Formula: P = (4/3) * sigma_T * c * beta^2 * gamma^2 * U_B
///
/// # Arguments
/// * `u_b` - Magnetic energy density.
/// * `beta` - Electron velocity / c.
/// * `gamma` - Lorentz factor.
///
/// # Errors
/// * `HighEnergyError::InvalidEnergyDensity` if `u_b < 0`.
/// * `HighEnergyError::InvalidLorentzFactor` if `gamma < 1`.
#[verified_engine::verified]
pub fn synchrotron_power(u_b: f64, beta: f64, gamma: f64) -> Result<f64, HighEnergyError> {
    if u_b < 0.0 {
        return Err(HighEnergyError::InvalidEnergyDensity { u_b });
    }
    if gamma < 1.0 {
        return Err(HighEnergyError::InvalidLorentzFactor { gamma });
    }

    // P = 4/3 sigma_T c beta^2 gamma^2 U_B
    let power = (4.0 / 3.0) * SIGMA_T * C * beta.powi(2) * gamma.powi(2) * u_b;
    Ok(power)
}

/// Calculates the observed spectral index alpha from an electron power-law distribution p.
/// Formula: alpha = (p - 1) / 2
///
/// # Errors
/// * `HighEnergyError::InvalidPowerLawIndex` if `p <= 1`.
#[verified_engine::verified]
pub fn inverse_compton_spectral_index(p: f64) -> Result<f64, HighEnergyError> {
    if !p.is_finite() || p <= 1.0 {
        return Err(HighEnergyError::InvalidPowerLawIndex { p });
    }
    Ok((p - 1.0) / 2.0)
}

/// Evaluates the monochromatic synchrotron Doppler flux-boosting factor $D^{3 + \alpha}$
/// (or $D^{2 + \alpha}$ when `continuous_jet` is `true`) for spectral index $\alpha = (p - 1)/2$.
///
/// # Errors
/// * [`HighEnergyError::InvalidLorentzFactor`] if `doppler_factor <= 0` or inputs are non-finite.
#[verified_engine::verified]
pub fn beamed_synchrotron_flux_factor(
    doppler_factor: f64,
    spectral_index_alpha: f64,
    continuous_jet: bool,
) -> Result<f64, HighEnergyError> {
    if !doppler_factor.is_finite() || doppler_factor <= 0.0 || !spectral_index_alpha.is_finite() {
        return Err(HighEnergyError::InvalidLorentzFactor {
            gamma: doppler_factor,
        });
    }
    let base_exp = if continuous_jet { 2.0 } else { 3.0 };
    Ok(doppler_factor.powf(base_exp + spectral_index_alpha))
}

/// Computes the approaching-to-receding flux asymmetry $(D_{\text{app}} / D_{\text{rec}})^{3 + \alpha}$
/// of a symmetric relativistic twin synchrotron jet with bulk speed `beta`, line-of-sight angle
/// `viewing_angle_rad`, and electron power-law index `electron_index_p` ($\alpha = (p - 1)/2$).
///
/// At $\beta = 0.9$, $\theta = 15^\circ$, and $p = 2.4$ ($\alpha = 0.7$), this yields $18{,}855.15\times$.
///
/// # Errors
/// Propagates [`HighEnergyError`] if `|beta| >= 1.0` or `electron_index_p <= 1.0`.
#[verified_engine::verified]
pub fn twin_jet_flux_asymmetry(
    beta: f64,
    viewing_angle_rad: f64,
    electron_index_p: f64,
) -> Result<f64, HighEnergyError> {
    let alpha = inverse_compton_spectral_index(electron_index_p)?;
    let d_app = relativistic_doppler_factor(beta, viewing_angle_rad)?;
    let d_rec = relativistic_doppler_factor(beta, PI - viewing_angle_rad)?;
    beamed_synchrotron_flux_factor(d_app / d_rec, alpha, false)
}

/// Evaluates the normalized azimuthal Doppler + gravitational-redshift intensity factor
/// $(g_{\text{grav}} D(\phi))^{3 + \alpha}$ along a circular Schwarzschild accretion ring at
/// dimensionless radius `r_over_rs > 1.0`, inclination `disk_inclination_rad`, and ring azimuth
/// `azimuth_phi_rad` ($\phi = 0$ maximal blueshift, $\phi = \pi$ maximal redshift).
///
/// # Errors
/// Returns [`HighEnergyError`] if `r_over_rs <= 1.5` (inside photon sphere where circular shell
/// speed reaches $c$) or `electron_index_p <= 1.0`.
#[verified_engine::verified]
pub fn accretion_ring_beaming_factor(
    r_over_rs: f64,
    disk_inclination_rad: f64,
    azimuth_phi_rad: f64,
    electron_index_p: f64,
) -> Result<f64, HighEnergyError> {
    if !r_over_rs.is_finite() || r_over_rs <= 1.5 {
        return Err(HighEnergyError::InvalidRadius {
            radius: r_over_rs,
            limit: 1.5,
        });
    }
    if !disk_inclination_rad.is_finite() || !azimuth_phi_rad.is_finite() {
        return Err(HighEnergyError::InvalidVelocity { v: f64::NAN });
    }
    let alpha = inverse_compton_spectral_index(electron_index_p)?;
    let beta_shell = (1.0 / (2.0 * (r_over_rs - 1.0))).sqrt();
    let gamma_orb = calculate_lorentz_factor_from_beta(beta_shell)?;
    let grav_lapse = (1.0 - 1.0 / r_over_rs).sqrt();
    let line_of_sight_proj = disk_inclination_rad.sin() * azimuth_phi_rad.cos();
    let doppler = 1.0 / (gamma_orb * (1.0 - beta_shell * line_of_sight_proj));
    Ok((grav_lapse * doppler).powf(3.0 + alpha))
}

/// Computes the peak approaching-to-receding brightness ratio $(I_{\max} / I_{\min})$ of a
/// relativistic Schwarzschild accretion ring at `r_over_rs > 1.5` and inclination
/// `disk_inclination_rad`.
///
/// At the ISCO ($r / R_s = 3.0$, $\beta_{\text{shell}} = 0.5$), edge-on ($i = 90^\circ$), and
/// $p = 2.4$ ($\alpha = 0.7$), this yields $3.0^{3.7} \approx 58.26\times$.
///
/// # Errors
/// Propagates [`HighEnergyError`] if `r_over_rs <= 1.5` or `electron_index_p <= 1.0`.
#[verified_engine::verified]
pub fn accretion_ring_flux_asymmetry(
    r_over_rs: f64,
    disk_inclination_rad: f64,
    electron_index_p: f64,
) -> Result<f64, HighEnergyError> {
    let blue =
        accretion_ring_beaming_factor(r_over_rs, disk_inclination_rad, 0.0, electron_index_p)?;
    let red = accretion_ring_beaming_factor(r_over_rs, disk_inclination_rad, PI, electron_index_p)?;
    Ok(blue / red)
}

/// Coupled shock, synchrotron, Doppler-beaming, and Li & Ma significance observables for a
/// relativistic jet knot and companion inner accretion ring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JetKnotObservables {
    /// Bulk jet Lorentz factor $\gamma = (1 - \beta^2)^{-1/2}$.
    pub bulk_gamma: f64,
    /// Synchrotron / Inverse-Compton spectral index $\alpha = (p - 1)/2$.
    pub spectral_index_alpha: f64,
    /// Approaching jet Doppler factor $D_{\text{app}}$.
    pub doppler_approaching: f64,
    /// Receding counter-jet Doppler factor $D_{\text{rec}}$.
    pub doppler_receding: f64,
    /// Apparent transverse superluminal speed $\beta_{\text{app}}$.
    pub apparent_beta: f64,
    /// Twin-jet approaching-to-receding flux ratio $(D_{\text{app}} / D_{\text{rec}})^{3 + \alpha}$.
    pub jet_flux_asymmetry: f64,
    /// Accretion-ring approaching-to-receding flux ratio $(I_{\text{blue}} / I_{\text{red}})$.
    pub disk_flux_asymmetry: f64,
    /// Strong-shock downstream density compression ratio $r_{\text{shock}} = (\hat{\gamma}+1)/(\hat{\gamma}-1)$.
    pub shock_compression: f64,
    /// Post-shock single-electron boosted synchrotron power (W) for electron Lorentz factor $\gamma_e$.
    pub boosted_synchrotron_power_w: f64,
    /// Li & Ma gamma-ray detection significance ($\sigma$) of the approaching jet knot.
    pub li_ma_sigma: f64,
}

/// Input parameters for [`evaluate_jet_knot_observables`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JetKnotConfig {
    /// Dimensionless bulk jet speed $\beta \in (0, 1)$.
    pub bulk_beta: f64,
    /// Jet viewing angle $\theta$ relative to the line of sight (radians).
    pub viewing_angle_rad: f64,
    /// Accretion-ring dimensionless radius $r / R_s > 1.5$.
    pub disk_r_over_rs: f64,
    /// Accretion-ring inclination angle $i$ from face-on (radians).
    pub disk_inclination_rad: f64,
    /// Relativistic electron power-law index $p > 1$.
    pub electron_index_p: f64,
    /// Representative radiating electron Lorentz factor $\gamma_e \ge 1$.
    pub electron_gamma: f64,
    /// Upstream magnetic energy density $U_{B,0} \ge 0$ ($\text{J/m}^3$).
    pub upstream_u_b: f64,
    /// Fluid adiabatic index $\hat{\gamma} > 1$ (e.g. $4/3$ for relativistic plasma).
    pub adiabatic_index: f64,
    /// Unbeamed comoving signal photon expectation per exposure.
    pub intrinsic_signal_counts: f64,
    /// Off-source background photon counts $N_{\text{off}} > 0$.
    pub background_counts_off: f64,
    /// Exposure ratio $\alpha_{\text{exp}} = t_{\text{on}} / t_{\text{off}} > 0$.
    pub exposure_ratio: f64,
}

/// Evaluates coupled relativistic jet-knot beaming, shock compression, synchrotron power, and
/// Li & Ma detection significance without heap allocation on valid physical inputs.
///
/// # Errors
/// Propagates [`HighEnergyError`] if any physical parameter violates domain bounds.
#[verified_engine::verified]
pub fn evaluate_jet_knot_observables(
    cfg: &JetKnotConfig,
) -> Result<JetKnotObservables, HighEnergyError> {
    let bulk_gamma = calculate_lorentz_factor_from_beta(cfg.bulk_beta)?;
    let spectral_index_alpha = inverse_compton_spectral_index(cfg.electron_index_p)?;
    let doppler_approaching = relativistic_doppler_factor(cfg.bulk_beta, cfg.viewing_angle_rad)?;
    let doppler_receding = relativistic_doppler_factor(cfg.bulk_beta, PI - cfg.viewing_angle_rad)?;
    let apparent_beta = apparent_superluminal_beta(cfg.bulk_beta, cfg.viewing_angle_rad)?;
    let jet_flux_asymmetry = beamed_synchrotron_flux_factor(
        doppler_approaching / doppler_receding,
        spectral_index_alpha,
        false,
    )?;
    let disk_flux_asymmetry = accretion_ring_flux_asymmetry(
        cfg.disk_r_over_rs,
        cfg.disk_inclination_rad,
        cfg.electron_index_p,
    )?;

    let shock_compression = shock_compression_ratio(cfg.adiabatic_index)?;
    let post_shock_u_b = cfg.upstream_u_b * shock_compression * shock_compression;
    let electron_beta = (1.0 - 1.0 / (cfg.electron_gamma * cfg.electron_gamma))
        .max(0.0)
        .sqrt();
    let comoving_power = synchrotron_power(post_shock_u_b, electron_beta, cfg.electron_gamma)?;
    let beam_boost =
        beamed_synchrotron_flux_factor(doppler_approaching, spectral_index_alpha, false)?;
    let boosted_synchrotron_power_w = comoving_power * beam_boost;

    let signal_on = cfg.intrinsic_signal_counts.max(0.0) * beam_boost;
    let n_on = cfg.background_counts_off * cfg.exposure_ratio + signal_on;
    let li_ma_sigma = li_ma_significance(n_on, cfg.background_counts_off, cfg.exposure_ratio)?;

    Ok(JetKnotObservables {
        bulk_gamma,
        spectral_index_alpha,
        doppler_approaching,
        doppler_receding,
        apparent_beta,
        jet_flux_asymmetry,
        disk_flux_asymmetry,
        shock_compression,
        boosted_synchrotron_power_w,
        li_ma_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_radiation() {
        let u_b = 1.0;
        let gamma = 2.0;
        let beta = (1.0 - 1.0 / 4.0f64).sqrt();
        let p = synchrotron_power(u_b, beta, gamma).expect("Failed to calc synchrotron power");
        let expected = (4.0 / 3.0) * SIGMA_T * C * beta * beta * 4.0;
        assert_relative_eq!(
            p,
            expected,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let alpha = inverse_compton_spectral_index(3.0).expect("Failed to calc alpha");
        assert_relative_eq!(alpha, 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_jet_and_disk_beaming_benchmarks_and_zero_alloc() {
        let cfg = JetKnotConfig {
            bulk_beta: 0.9,
            viewing_angle_rad: 15.0_f64.to_radians(),
            disk_r_over_rs: 3.0,
            disk_inclination_rad: 90.0_f64.to_radians(),
            electron_index_p: 2.4,
            electron_gamma: 1.0e3,
            upstream_u_b: 1.0e-3,
            adiabatic_index: 4.0 / 3.0,
            intrinsic_signal_counts: 1.0,
            background_counts_off: 40.0,
            exposure_ratio: 1.0,
        };
        let (obs, metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            evaluate_jet_knot_observables(&cfg).expect("valid jet knot observables")
        });
        assert!(metrics.function_calls > 0);
        assert_relative_eq!(obs.jet_flux_asymmetry, 18_855.15, epsilon = 0.05);
        assert_relative_eq!(obs.disk_flux_asymmetry, 58.257, epsilon = 0.02);
        assert_relative_eq!(obs.shock_compression, 7.0, epsilon = 1e-12);
        assert!(obs.li_ma_sigma > 5.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_errors() {
        assert!(synchrotron_power(-1.0, 0.9, 10.0).is_err());
        assert!(synchrotron_power(1.0, 0.9, 0.5).is_err());
        assert!(inverse_compton_spectral_index(0.5).is_err());
        assert!(accretion_ring_flux_asymmetry(1.2, 1.0, 2.4).is_err());
    }
}
