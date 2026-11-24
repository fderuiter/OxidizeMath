//! Federated `astrophysics_mystery_halo_fit` inverse challenge, 16-point observed dwarf-galaxy
//! rotation curve targets, zero-allocation residual evaluator, and `CalloutManager` sync.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rotation_curve_tool::{RotationCurveExplorer, RotationCurveState};
use super::{clamp_finite, HaloKindChoice};
use crate::framework::{ChallengeEvaluation, ChallengeMetadata, InteractiveTool};
use math_explorer::physics::astrophysics::{
    decompose_rotation_curve, BaryonicDiskConfig, DwarfMorphology, HaloProfile,
};

/// Canonical challenge ID for fitting the mystery irregular dwarf-galaxy dark-matter halo.
pub const CHALLENGE_ASTRO_MYSTERY_HALO_FIT: &str = "astrophysics_mystery_halo_fit";

/// Number of observed HI + optical circular velocity data points across the mystery dwarf galaxy.
pub const NUM_MYSTERY_TARGETS: usize = 16;
/// Degrees of freedom (`16` observations minus `3` fitted halo parameters: `M200`, `r_s`, `c`).
pub const MYSTERY_DOF: f64 = 13.0;

/// Locked galactocentric distance in Mpc (`14.5 Mpc`).
pub const LOCKED_DISTANCE_MPC: f64 = 14.5;
/// Locked fallback V-band absolute magnitude (`-15.8 mag`).
pub const LOCKED_ABS_MAG_V: f64 = -15.8;
/// Locked HI gas mass fraction (`0.62`).
pub const LOCKED_GAS_FRACTION: f64 = 0.62;
/// Locked Kuzmin stellar disk scale length in kpc (`0.85 kpc`).
pub const LOCKED_DISK_SCALE_KPC: f64 = 0.85;
/// Locked radial plot extent in kpc (`13.0 kpc`).
pub const LOCKED_MAX_RADIUS_KPC: f64 = 13.0;

/// Maximum RMSE (`km/s`) permitted to pass `astrophysics_mystery_halo_fit`.
pub const MAX_PASSING_RMSE_KMS: f64 = 2.40;
/// Maximum reduced chi-squared ($\chi^2_\nu$) permitted to pass `astrophysics_mystery_halo_fit`.
pub const MAX_PASSING_REDUCED_CHI_SQ: f64 = 1.35;

/// Single galactocentric circular-velocity measurement with $1\sigma$ uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityObservation {
    /// Galactocentric radius $r$ in kpc.
    pub radius_kpc: f64,
    /// Observed circular velocity $v_{\text{obs}}(r)$ in km/s.
    pub v_obs_kms: f64,
    /// $1\sigma$ observational velocity uncertainty $\sigma_v(r)$ in km/s.
    pub sigma_kms: f64,
}

/// 16 empirical-scale dwarf-galaxy HI rotation curve measurements exhibiting a constant-density inner core.
pub const MYSTERY_DWARF_TARGETS: [VelocityObservation; NUM_MYSTERY_TARGETS] = [
    VelocityObservation {
        radius_kpc: 0.55,
        v_obs_kms: 15.74,
        sigma_kms: 2.10,
    },
    VelocityObservation {
        radius_kpc: 0.95,
        v_obs_kms: 25.74,
        sigma_kms: 2.10,
    },
    VelocityObservation {
        radius_kpc: 1.40,
        v_obs_kms: 31.45,
        sigma_kms: 2.20,
    },
    VelocityObservation {
        radius_kpc: 1.90,
        v_obs_kms: 39.28,
        sigma_kms: 2.20,
    },
    VelocityObservation {
        radius_kpc: 2.45,
        v_obs_kms: 44.68,
        sigma_kms: 2.30,
    },
    VelocityObservation {
        radius_kpc: 3.05,
        v_obs_kms: 46.60,
        sigma_kms: 2.40,
    },
    VelocityObservation {
        radius_kpc: 3.70,
        v_obs_kms: 50.70,
        sigma_kms: 2.40,
    },
    VelocityObservation {
        radius_kpc: 4.40,
        v_obs_kms: 50.96,
        sigma_kms: 2.50,
    },
    VelocityObservation {
        radius_kpc: 5.15,
        v_obs_kms: 54.36,
        sigma_kms: 2.60,
    },
    VelocityObservation {
        radius_kpc: 5.95,
        v_obs_kms: 53.54,
        sigma_kms: 2.70,
    },
    VelocityObservation {
        radius_kpc: 6.80,
        v_obs_kms: 55.92,
        sigma_kms: 2.80,
    },
    VelocityObservation {
        radius_kpc: 7.70,
        v_obs_kms: 53.71,
        sigma_kms: 2.90,
    },
    VelocityObservation {
        radius_kpc: 8.65,
        v_obs_kms: 55.17,
        sigma_kms: 3.00,
    },
    VelocityObservation {
        radius_kpc: 9.65,
        v_obs_kms: 53.16,
        sigma_kms: 3.10,
    },
    VelocityObservation {
        radius_kpc: 10.75,
        v_obs_kms: 54.61,
        sigma_kms: 3.20,
    },
    VelocityObservation {
        radius_kpc: 11.90,
        v_obs_kms: 52.36,
        sigma_kms: 3.30,
    },
];

/// Evaluates `(rmse_kms, reduced_chi_sq)` into a stack-resident residual buffer with zero heap allocations.
#[must_use]
pub fn evaluate_mystery_residuals_into(
    baryons: &BaryonicDiskConfig,
    halo: &HaloProfile,
    residuals_out: &mut [f64; NUM_MYSTERY_TARGETS],
) -> (f64, f64) {
    let mut sum_sq = 0.0_f64;
    let mut chi_sq = 0.0_f64;
    for (idx, obs) in MYSTERY_DWARF_TARGETS.iter().enumerate() {
        let v_model = decompose_rotation_curve(
            math_scalars::Kiloparsecs::new(obs.radius_kpc),
            baryons,
            halo,
        )
        .v_circ_kms;
        let dv = v_model - obs.v_obs_kms;
        residuals_out[idx] = dv;
        sum_sq += dv * dv;
        let sigma = obs.sigma_kms.max(0.1);
        chi_sq += (dv / sigma).powi(2);
    }
    let rmse = (sum_sq / (NUM_MYSTERY_TARGETS as f64)).sqrt();
    let red_chi_sq = chi_sq / MYSTERY_DOF;
    (rmse, red_chi_sq)
}

impl RotationCurveExplorer {
    /// Returns `true` when `astrophysics_mystery_halo_fit` locks baryonic distance and disk scaling.
    #[must_use]
    pub(crate) fn is_challenge_locked(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT)
    }

    /// Restores canonical locked baryonic and radial bounds on `state`.
    pub(crate) fn enforce_challenge_locks_on_state(s: &mut RotationCurveState) {
        s.distance_mpc = LOCKED_DISTANCE_MPC;
        s.abs_magnitude_v = LOCKED_ABS_MAG_V;
        s.use_absolute_magnitude = false;
        s.morphology = DwarfMorphology::TypeCode10;
        s.gas_fraction = LOCKED_GAS_FRACTION;
        s.disk_scale_kpc = LOCKED_DISK_SCALE_KPC;
        s.max_radius_kpc = LOCKED_MAX_RADIUS_KPC;
    }

    /// Activates or clears `astrophysics_mystery_halo_fit` on [`RotationCurveExplorer`].
    pub(crate) fn set_astro_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.recalculate();
                Ok(())
            }
            Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT) => {
                self.active_challenge = Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT);
                Self::enforce_challenge_locks_on_state(&mut self.cfg);
                self.recalculate();
                Ok(())
            }
            Some(other) => Err(format!(
                "Unsupported challenge '{other}' for RotationCurveExplorer"
            )),
        }
    }

    /// Scores live `rmse_kms` and `reduced_chi_sq` across the 16 dwarf-galaxy velocity targets.
    #[must_use]
    pub(crate) fn eval_astro_challenge(&self) -> Option<ChallengeEvaluation> {
        self.active_challenge?;
        Some(score_mystery_halo_fit(
            self.rmse_kms,
            self.reduced_chi_sq,
            self.cfg.halo_kind,
        ))
    }

    /// Synchronizes `CalloutManager` residual tags for fit summary and peak standardized residual.
    pub(crate) fn sync_rotation_callouts(&mut self) {
        let mut worst_idx = 0_usize;
        let mut worst_norm = -1.0_f64;
        for (idx, (&dv, obs)) in self
            .target_residuals_kms
            .iter()
            .zip(MYSTERY_DWARF_TARGETS.iter())
            .enumerate()
        {
            let norm = (dv / obs.sigma_kms.max(0.1)).abs();
            if norm > worst_norm {
                worst_norm = norm;
                worst_idx = idx;
            }
        }
        let worst_obs = MYSTERY_DWARF_TARGETS[worst_idx];
        let worst_dv = self.target_residuals_kms[worst_idx];
        self.callout_manager.set_auto_callout(
            "astro_worst_residual",
            [worst_obs.radius_kpc, worst_obs.v_obs_kms],
            format!(
                "Peak Δv = {:+.1} km/s @ {:.2} kpc",
                worst_dv, worst_obs.radius_kpc
            ),
            [0.0_f32, -20.0_f32],
        );

        let anchor_y = self.samples.get(76).map_or(52.0, |s| s.v_circ_kms);
        self.callout_manager.set_auto_callout(
            "astro_fit_summary",
            [9.8, anchor_y],
            format!(
                "RMSE = {:.2} km/s, χ²_ν = {:.2}",
                self.rmse_kms, self.reduced_chi_sq
            ),
            [0.0_f32, 20.0_f32],
        );
    }
}

/// Pure scorer mapping `rmse_kms` and `reduced_chi_sq` to a [`ChallengeEvaluation`].
#[must_use]
pub fn score_mystery_halo_fit(
    rmse_kms: f64,
    reduced_chi_sq: f64,
    halo_kind: HaloKindChoice,
) -> ChallengeEvaluation {
    let passed = rmse_kms <= MAX_PASSING_RMSE_KMS && reduced_chi_sq <= MAX_PASSING_REDUCED_CHI_SQ;
    let score = if passed {
        let chi_ratio = (reduced_chi_sq / MAX_PASSING_REDUCED_CHI_SQ).clamp(0.0, 1.0);
        100.0 - chi_ratio * 9.0
    } else {
        let excess = (reduced_chi_sq - MAX_PASSING_REDUCED_CHI_SQ).max(0.0);
        (89.0 / (1.0 + 0.14 * excess)).clamp(0.0, 89.0)
    };
    ChallengeEvaluation::new(
        score,
        passed,
        "Reduced χ²_ν",
        reduced_chi_sq,
        format!(
            "{}: χ²_ν = {reduced_chi_sq:.2} (<= {MAX_PASSING_REDUCED_CHI_SQ:.2}), RMSE = {rmse_kms:.2} km/s (<= {MAX_PASSING_RMSE_KMS:.2} km/s)",
            halo_kind.label()
        ),
    )
}

fn generate_astro_mystery_baseline() -> String {
    let mut tool = RotationCurveExplorer::default();
    RotationCurveExplorer::enforce_challenge_locks_on_state(&mut tool.cfg);
    tool.cfg.halo_kind = HaloKindChoice::NfwCusp;
    tool.cfg.halo_log10_mass = 9.5;
    tool.cfg.halo_scale_kpc = 4.2;
    tool.cfg.halo_concentration = 8.5;
    tool.cfg.compare_counterpart = true;
    tool.save_state().unwrap_or_default()
}

fn clamp_astro_mystery_state(raw_json: &str) -> Result<String, String> {
    let mut val: serde_json::Value = serde_json::from_str(raw_json)
        .map_err(|e| format!("Invalid RotationCurveExplorer challenge JSON: {e}"))?;
    let obj = val
        .as_object_mut()
        .ok_or_else(|| "RotationCurveExplorer challenge state must be a JSON object".to_string())?;
    obj.remove("snapshot_manager");
    obj.insert(
        "distance_mpc".to_string(),
        serde_json::json!(LOCKED_DISTANCE_MPC),
    );
    obj.insert(
        "abs_magnitude_v".to_string(),
        serde_json::json!(LOCKED_ABS_MAG_V),
    );
    obj.insert(
        "use_absolute_magnitude".to_string(),
        serde_json::json!(false),
    );
    obj.insert("morphology".to_string(), serde_json::json!("TypeCode10"));
    obj.insert(
        "gas_fraction".to_string(),
        serde_json::json!(LOCKED_GAS_FRACTION),
    );
    obj.insert(
        "disk_scale_kpc".to_string(),
        serde_json::json!(LOCKED_DISK_SCALE_KPC),
    );
    obj.insert(
        "max_radius_kpc".to_string(),
        serde_json::json!(LOCKED_MAX_RADIUS_KPC),
    );
    for (key, lo, hi, fb) in [
        ("halo_log10_mass", 8.0, 12.0, 9.5),
        ("halo_scale_kpc", 0.3, 10.0, 4.2),
        ("halo_concentration", 4.0, 25.0, 8.5),
    ] {
        if let Some(v) = obj.get(key).and_then(serde_json::Value::as_f64) {
            obj.insert(
                key.to_string(),
                serde_json::json!(clamp_finite(v, lo, hi, fb)),
            );
        }
    }

    let sanitized = serde_json::to_string(&val).map_err(|e| e.to_string())?;
    let mut tool = RotationCurveExplorer::default();
    tool.set_astro_challenge(Some(CHALLENGE_ASTRO_MYSTERY_HALO_FIT))?;
    tool.try_load_state(&sanitized)?;
    tool.save_state()
        .ok_or_else(|| "Failed to serialize clamped RotationCurveExplorer state".to_string())
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_ASTRO_MYSTERY_HALO_FIT,
        title: "Core vs. Cusp: Mystery Dwarf Galaxy Halo Fit",
        description: "Fit halo family (BurkertCore vs. NfwCusp), virial mass M200, scale radius r_s, and concentration c to 16 observed HI dwarf-galaxy circular velocities with locked baryons.",
        category: "Galactic Dynamics",
        icon: "🌌",
        domain: "astrophysics",
        tool_name: "RotationCurveExplorer",
        locked_inputs: "TypeCode10 dwarf, D = 14.5 Mpc, f_gas = 0.62, R_d = 0.85 kpc, r_max = 13.0 kpc",
        tunable_budget: "Halo family, log10(M200/M_sun) ∈ [8, 12], r_s ∈ [0.3, 10] kpc, c ∈ [4, 25]",
        target_summary: "Reduced chi-squared χ²_ν <= 1.35 and RMSE <= 2.40 km/s across 16 targets",
        baseline_state_generator: generate_astro_mystery_baseline,
        clamp_locked_state: clamp_astro_mystery_state,
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]
