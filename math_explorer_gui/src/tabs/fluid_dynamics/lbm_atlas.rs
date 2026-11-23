//! Zero-allocation BGK / Reynolds regime cartographer for [`super::lattice_boltzmann::LbmUnified`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::lattice_boltzmann::{LbmUnified, LBM_INLET_U};
use super::lbm_fairing::LBM_REF_DIAMETER;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Supported 2D parameter atlas axis pairs for [`LbmUnified`].
pub const LBM_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("viscosity", "inlet_velocity")];

const VISC_MIN: f64 = 0.00333;
const VISC_MAX: f64 = 0.5;
const INLET_U_MIN: f64 = 0.015;
const INLET_U_MAX: f64 = 0.185;
const KARMAN_RE_CRIT: f64 = 47.0;
const CREEPING_RE_MAX: f64 = 5.0;
const MACH_LIMIT: f64 = 0.28;
const TAU_MIN_SAFE: f64 = 0.515;

/// Stack-resident zero-allocation cartographer for D2Q9 BGK wind-tunnel regimes.
pub struct LbmCartographer {
    axes: AtlasAxisPair,
    char_length: f64,
}

impl LbmCartographer {
    /// Builds a cartographer for `("viscosity", "inlet_velocity")` given current [`LbmUnified`] parameters.
    #[must_use]
    pub fn from_params(
        params: &HashMap<String, f64>,
        axis_keys: (&'static str, &'static str),
    ) -> Option<Self> {
        if axis_keys != ("viscosity", "inlet_velocity") {
            return None;
        }
        let theory = <LbmUnified as crate::async_sim::unified::UnifiedModel>::parameters();
        let profile = params.get("naca_profile").copied().unwrap_or(0.0);
        let char_length = if profile >= 1.0 {
            24.0
        } else {
            LBM_REF_DIAMETER
        };
        Some(Self {
            axes: AtlasAxisPair {
                x: AtlasAxisSpec::from_theory(
                    &theory,
                    "viscosity",
                    "Kinematic Viscosity ν (lu²/ts)",
                    ParameterConstraint {
                        min: VISC_MIN,
                        max: VISC_MAX,
                        step: 0.005,
                    },
                ),
                y: AtlasAxisSpec::from_theory(
                    &theory,
                    "inlet_velocity",
                    "Inlet Velocity U (lu/ts)",
                    ParameterConstraint {
                        min: INLET_U_MIN,
                        max: INLET_U_MAX,
                        step: 0.005,
                    },
                ),
            },
            char_length,
        })
    }
}

impl RegimeCartographer for LbmCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let nu = p1.max(1e-6);
        let u_in = p2.max(0.0);
        let tau = 3.0 * nu + 0.5;
        let mach = 3.0_f64.sqrt() * u_in;
        let re = (u_in * self.char_length) / nu;
        let regime = if tau < TAU_MIN_SAFE || mach > MACH_LIMIT {
            3
        } else if re < CREEPING_RE_MAX {
            0
        } else if re < KARMAN_RE_CRIT {
            1
        } else {
            2
        };
        AtlasCell {
            regime,
            primary: re,
            secondary: tau,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = (cell.primary.ln_1p() / 6.0).clamp(0.0, 1.0);
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        match regime {
            0 => "Creeping Attached Flow (Re < 5)",
            1 => "Steady Twin Wake Bubble (5 ≤ Re < 47)",
            2 => "Kármán Vortex Street (Re ≥ 47)",
            _ => "BGK Under-Relaxed / Compressible Limit",
        }
    }

    fn primary_metric_label(&self) -> &'static str {
        "Reynolds Number Re"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        format!(
            "ν = {p1:.4}, U_in = {p2:.3}\nRegime: {}\nRe = {:.1} | τ = {:.3}",
            self.regime_label(cell.regime),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let mut karman_pts = Vec::with_capacity(48);
        let steps = 47;
        for i in 0..=steps {
            let frac = (i as f64) / (steps as f64);
            let nu = self.axes.x.constraint.min + frac * self.axes.x.span();
            let u_crit = (KARMAN_RE_CRIT * nu) / self.char_length;
            if (self.axes.y.constraint.min..=self.axes.y.constraint.max).contains(&u_crit) {
                karman_pts.push([nu, u_crit]);
            }
        }
        (karman_pts.len() >= 2).then_some(("Kármán Shedding Onset (Re = 47)", karman_pts))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        (0..4_u8)
            .map(|r| {
                (
                    self.regime_label(r),
                    atlas_regime_color(r, 4, 0.75, high_contrast),
                )
            })
            .collect()
    }
}

#[must_use]
pub(crate) fn resolve_inlet_velocity(params: &HashMap<String, f64>) -> f64 {
    params
        .get("inlet_velocity")
        .copied()
        .filter(|u| u.is_finite() && (0.005..=0.35).contains(u))
        .unwrap_or(LBM_INLET_U)
}
pub(super) fn evaluate_lbm_observable(
    params: &HashMap<String, f64>,
) -> Option<(&'static str, f64)> {
    use super::lbm_fairing::{LBM_INLET_COLS, LBM_RHO_0};
    use math_explorer::physics::fluid_dynamics::lattice_boltzmann::{
        compute_drag_coefficient, LatticeBoltzmannD2Q9,
    };

    let raw_visc = params.get("viscosity").copied().unwrap_or(0.02);
    if !raw_visc.is_finite() {
        return None;
    }
    let viscosity = raw_visc.clamp(0.002, 0.5);
    let u_in = resolve_inlet_velocity(params);
    let width = 32_usize;
    let height = 16_usize;
    let tau = 3.0 * viscosity + 0.5;
    let mut probe = LatticeBoltzmannD2Q9::new(width, height, tau);
    let cx = 10_i32;
    let cy = 8_i32;
    let radius = 2_i32;
    for y in 0..height {
        for x in 0..width {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            if dx * dx + dy * dy <= radius * radius {
                probe.set_obstacle(x, y, true);
            }
        }
    }
    probe.set_inlet(0, 0, LBM_INLET_COLS, height, u_in, 0.0);
    for _ in 0..20 {
        probe.set_inlet(0, 0, LBM_INLET_COLS, height, u_in, 0.0);
        probe.step();
    }
    let (f_drag, _) = probe.hydrodynamic_forces();
    let c_d = compute_drag_coefficient(f_drag, LBM_RHO_0, u_in, 5.0);
    c_d.is_finite().then_some(("Drag Coefficient (C_D)", c_d))
}
