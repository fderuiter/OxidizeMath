//! Zero-allocation Onsager/Yang 2D Ising order-parameter and spatial evolutionary-dilemma evaluators.

/// Exact Onsager denominator $\ln(1 + \sqrt{2})$.
const LN_ONE_PLUS_SQRT2: f64 = 0.881_373_587_019_543;

/// Exact Onsager critical temperature $T_c(J) = \frac{2|J|}{\ln(1 + \sqrt{2})}$.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn onsager_critical_temp(j_coupling: f64) -> f64 {
    if !j_coupling.is_finite() {
        return 0.0;
    }
    2.0 * j_coupling.abs() / LN_ONE_PLUS_SQRT2
}

/// Discrete phase and spatial-dilemma regimes on the 2D square lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IsingAtlasRegime {
    /// Disordered paramagnetic ($T > T_c$) or Prisoner's Dilemma defector freeze ($2R + 2S < T + 3P$).
    DisorderedOrDefectorFreeze = 0,
    /// Critical Curie crossover ($T \approx T_c$) or Stag-Hunt bistable coordination.
    CriticalOrBistable = 1,
    /// Ferromagnetic ordered ($J > 0, T < T_c$) or Spatial Reciprocity PD cluster survival.
    FerroOrSpatialReciprocity = 2,
    /// Antiferromagnetic checkerboard ($J < 0, T < T_c$) or Snowdrift/Harmony cooperation.
    AntiferroOrCoexistence = 3,
}

impl IsingAtlasRegime {
    /// Converts raw `u8` code into [`Self`].
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::DisorderedOrDefectorFreeze,
            1 => Self::CriticalOrBistable,
            2 => Self::FerroOrSpatialReciprocity,
            _ => Self::AntiferroOrCoexistence,
        }
    }

    /// Human-readable label conditioned on whether the `(T, S)` dilemma slice is active.
    #[inline]
    #[must_use]
    pub const fn label(self, is_game_slice: bool) -> &'static str {
        if is_game_slice {
            match self {
                Self::DisorderedOrDefectorFreeze => "PD Defector Freeze (2R + 2S < T + 3P)",
                Self::CriticalOrBistable => "Stag-Hunt Coordination (T ≤ R, S < P)",
                Self::FerroOrSpatialReciprocity => {
                    "PD Spatial Reciprocity Wedge (2R + 2S ≥ T + 3P)"
                }
                Self::AntiferroOrCoexistence => "Harmony / Snowdrift Coexistence (S ≥ P)",
            }
        } else {
            match self {
                Self::DisorderedOrDefectorFreeze => "Paramagnetic Disordered (T > Tc)",
                Self::CriticalOrBistable => "Critical Curie Crossover (T ≈ Tc)",
                Self::FerroOrSpatialReciprocity => "Ferromagnetic Ordered (J > 0, T < Tc)",
                Self::AntiferroOrCoexistence => "Antiferromagnetic Checkerboard (J < 0, T < Tc)",
            }
        }
    }
}

/// Evaluates Yang (1952) spontaneous magnetization and smooth external-field crossover in `O(1)` stack ops.
///
/// # Examples
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::solid_state::{
///     IsingAtlasRegime, eval_ising_order_parameter, onsager_critical_temp,
/// };
///
/// let tc = onsager_critical_temp(1.0);
/// let (low_reg, m_low, _) = eval_ising_order_parameter(1.0, 1.0, 0.0);
/// let (high_reg, m_high, _) = eval_ising_order_parameter(4.0, 1.0, 0.0);
/// assert!((tc - 2.269185).abs() < 1e-5);
/// assert_eq!(low_reg, IsingAtlasRegime::FerroOrSpatialReciprocity);
/// assert!(m_low > 0.9 && m_high == 0.0);
/// assert_eq!(high_reg, IsingAtlasRegime::DisorderedOrDefectorFreeze);
/// ```
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn eval_ising_order_parameter(temp: f64, j: f64, h: f64) -> (IsingAtlasRegime, f64, f64) {
    let t = temp.max(1e-3);
    let abs_j = j.abs();
    let tc = onsager_critical_temp(j);
    if abs_j < 1e-6 {
        let m_para = (h / t).tanh().abs();
        return (IsingAtlasRegime::DisorderedOrDefectorFreeze, m_para, 0.0);
    }

    let yang_m0 = if t < tc {
        let sh = (2.0 * abs_j / t).min(50.0).sinh();
        (1.0 - sh.powi(-4)).max(0.0).powf(0.125)
    } else {
        0.0
    };

    let order = if j >= 0.0 {
        let mut m = if h.abs() < 1e-6 {
            yang_m0
        } else {
            h.signum() * yang_m0.max(0.25)
        };
        if h.abs() >= 1e-6 {
            let eff_scale = tc / 4.0;
            for _ in 0..8 {
                m = ((4.0 * eff_scale * m + h) / t).tanh();
            }
        }
        m.abs().max(yang_m0).clamp(0.0, 1.0)
    } else {
        let h_crit = (4.0 * abs_j).max(1e-6);
        let field_suppression = (1.0 - (h.abs() / h_crit).powi(2)).clamp(0.0, 1.0).sqrt();
        (yang_m0 * field_suppression).clamp(0.0, 1.0)
    };

    let band = (0.04 * tc).max(0.05);
    let regime = if (t - tc).abs() <= band {
        IsingAtlasRegime::CriticalOrBistable
    } else if t > tc {
        IsingAtlasRegime::DisorderedOrDefectorFreeze
    } else if j >= 0.0 {
        IsingAtlasRegime::FerroOrSpatialReciprocity
    } else {
        IsingAtlasRegime::AntiferroOrCoexistence
    };
    (regime, order, tc)
}

/// Evaluates analytical `(T, S)` spatial evolutionary survival fraction and 0D well-mixed reference.
#[inline]
#[must_use]
#[verified_engine::verified]
pub fn eval_spatial_dilemma_cell(
    temptation: f64,
    sucker: f64,
    reward: f64,
    punishment: f64,
    noise_temp: f64,
) -> (IsingAtlasRegime, f64, f64) {
    let temp = noise_temp.max(1e-3);
    if temptation <= reward && sucker >= punishment {
        let advantage = ((reward - temptation) + (sucker - punishment)).max(0.0);
        let x_sp =
            (1.0 / (1.0 + (-2.0 * advantage / temp).clamp(-40.0, 40.0).exp())).clamp(0.5, 1.0);
        return (IsingAtlasRegime::AntiferroOrCoexistence, x_sp, 1.0);
    }
    if temptation > reward && sucker >= punishment {
        let num = (sucker - punishment).max(0.0);
        let den = ((temptation - reward) + (sucker - punishment)).max(1e-6);
        let x_ode = (num / den).clamp(0.0, 1.0);
        let x_sp = (x_ode * 0.88 + 0.08).clamp(0.05, 0.98);
        return (IsingAtlasRegime::AntiferroOrCoexistence, x_sp, x_ode);
    }
    if temptation <= reward && sucker < punishment {
        let coord_adv = (2.0 * reward + 2.0 * sucker) - (temptation + 3.0 * punishment);
        let x_sp = 1.0 / (1.0 + (-coord_adv / temp.max(0.05)).clamp(-40.0, 40.0).exp());
        let x_ode = if reward - temptation >= punishment - sucker {
            1.0
        } else {
            0.0
        };
        return (IsingAtlasRegime::CriticalOrBistable, x_sp, x_ode);
    }

    let cluster_margin = 2.0 * reward + 2.0 * sucker - (temptation + 3.0 * punishment);
    if cluster_margin >= 0.0 {
        let strength = (cluster_margin / (reward - punishment).abs().max(0.25)).clamp(0.0, 1.0);
        let thermal_damp = (-0.35 * temp).exp();
        let x_sp = ((0.22 + 0.72 * strength) * thermal_damp).clamp(0.05, 0.96);
        (IsingAtlasRegime::FerroOrSpatialReciprocity, x_sp, 0.0)
    } else {
        let leak =
            (0.04 * (cluster_margin / temp.max(0.04)).clamp(-40.0, 0.0).exp()).clamp(0.0, 0.04);
        (IsingAtlasRegime::DisorderedOrDefectorFreeze, leak, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_ising_order_parameter_and_spatial_dilemma_zero_alloc() {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((ferro, pd_reciprocity), _) = VerifiedEngine::run_verified(|| {
            (
                eval_ising_order_parameter(1.2, 1.0, 0.05),
                eval_spatial_dilemma_cell(1.3, -0.1, 1.0, 0.0, 0.5),
            )
        });
        assert_eq!(ferro.0, IsingAtlasRegime::FerroOrSpatialReciprocity);
        assert!(ferro.1 > 0.9);
        assert_eq!(
            pd_reciprocity.0,
            IsingAtlasRegime::FerroOrSpatialReciprocity
        );
        assert!(pd_reciprocity.1 > 0.2 && pd_reciprocity.2 == 0.0);
    }
}

// [cite:mmwave_radiotherapy_setup]
