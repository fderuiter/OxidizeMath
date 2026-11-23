#![cfg_attr(any(), verified(opt_out = "gui_tool"))]
// [cite:advanced_linear_algebra]

//! Zero-allocation 2D Kirchhoff-Love & Bessel modal parameter atlas and antinodal lock-in.

use std::collections::HashMap;
use std::f64::consts::TAU;

use eframe::egui;
use math_explorer::pure_math::analysis::pde::wave_2d::{
    Eigenmode2D, PinConstraint2D, Wave2DMedium, Wave2DParams, circular_drumhead_eigenfrequency_hz,
    evaluate_eigenmode_amplitude, populate_eigenmodes, square_plate_eigenfrequency_hz,
};
use scientific_metadata::theory::ParameterConstraint;

pub(crate) use super::lock_in::chladni_parameter_constraints;
use super::{
    ChladniWaveTankState, ClampLayout, FREQ_MAX, FREQ_MIN, GRID_SIZE, SPEED_MAX, SPEED_MIN,
    STIFFNESS_MAX, STIFFNESS_MIN, TankPreset,
};
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer, atlas_scalar_color,
};

/// Maximum surviving eigenmodes cached on the stack inside [`ChladniModalCartographer`].
pub(crate) const MAX_CHLADNI_MODES: usize = 24;
/// Normalized circular drumhead radius used by `TankPreset::CircularDrumhead`.
pub(crate) const DRUMHEAD_RADIUS_NORM: f64 = 0.45;

/// Supported `(x, y)` parameter axis pairs for [`ChladniModalCartographer`].
pub(crate) const CHLADNI_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("driver_frequency", "plate_stiffness"),
    ("driver_frequency", "wave_speed"),
    ("driver_x", "driver_y"),
];

pub(crate) fn default_chladni_atlas_axes() -> [String; 2] {
    [
        "driver_frequency".to_string(),
        "plate_stiffness".to_string(),
    ]
}

pub(crate) fn is_default_chladni_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "driver_frequency" && axes[1] == "plate_stiffness"
}

pub(crate) const fn is_false(flag: &bool) -> bool {
    !*flag
}

impl ClampLayout {
    pub(crate) const fn to_pin_constraint(self) -> PinConstraint2D {
        match self {
            Self::None => PinConstraint2D::None,
            Self::CenterPin => PinConstraint2D::CenterPin,
            Self::QuadSymmetry => PinConstraint2D::QuadSymmetry,
            Self::EdgeMidpoints => PinConstraint2D::EdgeMidpoints,
        }
    }
}

/// Discrete modal resonance regimes across frequency, dispersion, and driver coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum ChladniModalRegime {
    OffResonanceDecay = 0,
    NodalDeadZone = 1,
    ModalSkirt = 2,
    LockedStandingMode = 3,
}

impl ChladniModalRegime {
    #[inline]
    pub(crate) const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::OffResonanceDecay,
            1 => Self::NodalDeadZone,
            2 => Self::ModalSkirt,
            _ => Self::LockedStandingMode,
        }
    }

    #[inline]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::OffResonanceDecay => "Off-Resonance Decay",
            Self::NodalDeadZone => "Nodal Dead Zone (|Φ| ≈ 0)",
            Self::ModalSkirt => "Modal Resonance Skirt",
            Self::LockedStandingMode => "Locked Standing Eigenmode",
        }
    }
}

fn chladni_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    match key {
        "plate_stiffness" => AtlasAxisSpec::from_theory(
            theory_params,
            "plate_stiffness",
            "Plate Stiffness (D)",
            ParameterConstraint {
                min: STIFFNESS_MIN,
                max: STIFFNESS_MAX,
                step: 1.0e-5,
            },
        ),
        "wave_speed" => AtlasAxisSpec::from_theory(
            theory_params,
            "wave_speed",
            "Wave Speed (c)",
            ParameterConstraint {
                min: SPEED_MIN,
                max: SPEED_MAX,
                step: 0.01,
            },
        ),
        "driver_x" => AtlasAxisSpec::from_theory(
            theory_params,
            "driver_x",
            "Driver X (x_d)",
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        ),
        "driver_y" => AtlasAxisSpec::from_theory(
            theory_params,
            "driver_y",
            "Driver Y (y_d)",
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        ),
        _ => AtlasAxisSpec::from_theory(
            theory_params,
            "driver_frequency",
            "Driver Frequency (Hz)",
            ParameterConstraint {
                min: FREQ_MIN,
                max: FREQ_MAX,
                step: 0.001,
            },
        ),
    }
}

/// Stack-only 2D Lorentzian Kirchhoff-Bessel modal cartographer for [`super::ChladniWaveTankTool`].
#[derive(Clone, Debug)]
pub(crate) struct ChladniModalCartographer {
    axes: AtlasAxisPair,
    spatial_axes: bool,
    plate_active: bool,
    base_freq: f64,
    base_speed: f64,
    base_stiffness: f64,
    base_driver_x: f64,
    base_driver_y: f64,
    damping: f64,
    modes: [Eigenmode2D; MAX_CHLADNI_MODES],
    unit_wave_omega_sq: [f64; MAX_CHLADNI_MODES],
    unit_plate_omega_sq: [f64; MAX_CHLADNI_MODES],
    base_coupling: [f64; MAX_CHLADNI_MODES],
    mode_count: usize,
}

impl ChladniModalCartographer {
    #[must_use]
    pub(crate) fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        state: &ChladniWaveTankState,
        domain_len: f64,
    ) -> Self {
        let is_circ = state.preset == TankPreset::CircularDrumhead;
        let plate_active = state.medium == Wave2DMedium::BiharmonicPlate
            || axis_keys.0 == "plate_stiffness"
            || axis_keys.1 == "plate_stiffness";
        let probe = Wave2DParams {
            medium: if plate_active {
                Wave2DMedium::BiharmonicPlate
            } else {
                Wave2DMedium::AcousticWave
            },
            boundary: state.boundary,
            wave_speed: state.wave_speed.max(0.5),
            plate_stiffness: if plate_active {
                state.plate_stiffness.max(4.0e-4)
            } else {
                0.0
            },
            damping: state.damping,
        };
        let mut modes = [Eigenmode2D::default(); MAX_CHLADNI_MODES];
        let count = populate_eigenmodes(
            is_circ,
            GRID_SIZE,
            domain_len,
            DRUMHEAD_RADIUS_NORM,
            &probe,
            state.clamp_layout.to_pin_constraint(),
            &mut modes,
        );
        let (unit_wave_omega_sq, unit_plate_omega_sq, base_coupling) =
            precompute_mode_dispersion(&modes, count, is_circ, domain_len, state);
        Self {
            axes: AtlasAxisPair {
                x: chladni_axis_spec(theory_params, axis_keys.0),
                y: chladni_axis_spec(theory_params, axis_keys.1),
            },
            spatial_axes: axis_keys == ("driver_x", "driver_y"),
            plate_active,
            base_freq: state.driver_frequency,
            base_speed: state.wave_speed,
            base_stiffness: state.plate_stiffness,
            base_driver_x: state.driver_x,
            base_driver_y: state.driver_y,
            damping: state.damping.clamp(0.008, 5.0),
            modes,
            unit_wave_omega_sq,
            unit_plate_omega_sq,
            base_coupling,
            mode_count: count,
        }
    }

    #[inline]
    fn resolve(&self, p1: f64, p2: f64) -> (f64, f64, f64, f64, f64) {
        let mut f = self.base_freq;
        let mut c = self.base_speed;
        let mut d = if self.plate_active {
            self.base_stiffness
        } else {
            0.0
        };
        let (mut xd, mut yd) = (self.base_driver_x, self.base_driver_y);
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "driver_frequency" => f = v,
                "wave_speed" => c = v,
                "plate_stiffness" => d = v,
                "driver_x" => xd = v,
                "driver_y" => yd = v,
                _ => {}
            }
        }
        (f.max(1.0e-3), c.max(0.0), d.max(0.0), xd, yd)
    }
}

fn precompute_mode_dispersion(
    modes: &[Eigenmode2D; MAX_CHLADNI_MODES],
    count: usize,
    is_circ: bool,
    domain_len: f64,
    state: &ChladniWaveTankState,
) -> (
    [f64; MAX_CHLADNI_MODES],
    [f64; MAX_CHLADNI_MODES],
    [f64; MAX_CHLADNI_MODES],
) {
    let mut unit_w = [0.0; MAX_CHLADNI_MODES];
    let mut unit_p = [0.0; MAX_CHLADNI_MODES];
    let mut coup = [0.0; MAX_CHLADNI_MODES];
    let pw = Wave2DParams {
        medium: Wave2DMedium::AcousticWave,
        boundary: state.boundary,
        wave_speed: 1.0,
        plate_stiffness: 0.0,
        damping: 0.0,
    };
    let pp = Wave2DParams {
        medium: Wave2DMedium::BiharmonicPlate,
        boundary: state.boundary,
        wave_speed: 0.0,
        plate_stiffness: 1.0,
        damping: 0.0,
    };
    for i in 0..count {
        let m = &modes[i];
        let fw = if is_circ {
            circular_drumhead_eigenfrequency_hz(
                m.m,
                m.n,
                GRID_SIZE,
                domain_len,
                DRUMHEAD_RADIUS_NORM,
                &pw,
            )
        } else {
            square_plate_eigenfrequency_hz(m.m, m.n, GRID_SIZE, domain_len, &pw)
        };
        let fp = if is_circ {
            circular_drumhead_eigenfrequency_hz(
                m.m,
                m.n,
                GRID_SIZE,
                domain_len,
                DRUMHEAD_RADIUS_NORM,
                &pp,
            )
        } else {
            square_plate_eigenfrequency_hz(m.m, m.n, GRID_SIZE, domain_len, &pp)
        };
        unit_w[i] = (TAU * fw) * (TAU * fw);
        unit_p[i] = (TAU * fp) * (TAU * fp);
        coup[i] =
            evaluate_eigenmode_amplitude(m, state.driver_x, state.driver_y, DRUMHEAD_RADIUS_NORM)
                .abs();
    }
    (unit_w, unit_p, coup)
}

impl RegimeCartographer for ChladniModalCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        if self.mode_count == 0 {
            return AtlasCell::default();
        }
        let (freq, c, d, xd, yd) = self.resolve(p1, p2);
        let omega = TAU * freq;
        let mut total_gain = 0.0_f64;
        let mut best_term = -1.0_f64;
        let mut best_df = f64::INFINITY;
        let mut best_coup = 0.0_f64;
        let mut nearest_f_df = f64::INFINITY;
        let mut nearest_f_coup = 0.0_f64;

        for i in 0..self.mode_count {
            let w_sq =
                (c * c * self.unit_wave_omega_sq[i] + d * self.unit_plate_omega_sq[i]).max(0.0);
            let w_nu = w_sq.sqrt();
            let f_nu = w_nu / TAU;
            let df = (freq - f_nu).abs();
            let raw_c = if self.spatial_axes {
                evaluate_eigenmode_amplitude(&self.modes[i], xd, yd, DRUMHEAD_RADIUS_NORM).abs()
            } else {
                self.base_coupling[i]
            };
            if df < nearest_f_df {
                nearest_f_df = df;
                nearest_f_coup = raw_c;
            }
            let eff_c = if self.spatial_axes {
                raw_c
            } else {
                raw_c.max(0.55)
            };
            let dw = omega * omega - w_sq;
            let damp = self.damping * omega.max(w_nu).max(0.25);
            let term = (eff_c * eff_c * w_sq.max(0.05)) / (dw * dw + damp * damp).max(1.0e-12);
            total_gain += term;
            if term > best_term {
                best_term = term;
                best_df = df;
                best_coup = raw_c;
            }
        }
        classify_chladni_cell(
            self.spatial_axes,
            freq,
            total_gain,
            (best_df, best_coup),
            (nearest_f_df, nearest_f_coup),
        )
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let norm = ((1.0 + cell.primary.max(0.0)).ln() / 7.5).clamp(0.0, 1.0);
        atlas_scalar_color(norm, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        ChladniModalRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Modal Gain L(ω)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let extra = if self.spatial_axes {
            format!("Antinodal Coupling |Φ| = {:.2}", cell.secondary)
        } else {
            format!("Nearest Mode Detuning |Δf| = {:.3} Hz", cell.secondary)
        };
        format!(
            "{} = {:.4}, {} = {:.4}\nRegime: {}\nLorentzian Modal Gain = {:.2}  |  {}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            cell.primary,
            extra
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.mode_count == 0 || self.axes.x.key != "driver_frequency" {
            return None;
        }
        let idx = (self.mode_count / 3).min(self.mode_count - 1);
        let y_min = self.axes.y.constraint.min;
        let y_max = self.axes.y.constraint.max;
        let pts: Vec<[f64; 2]> = (0..=48)
            .filter_map(|s| {
                let y = y_min + (s as f64 / 48.0) * (y_max - y_min);
                let (_, c, d, _, _) = self.resolve(self.base_freq, y);
                let f = (c * c * self.unit_wave_omega_sq[idx] + d * self.unit_plate_omega_sq[idx])
                    .max(0.0)
                    .sqrt()
                    / TAU;
                (self.axes.x.constraint.min..=self.axes.x.constraint.max)
                    .contains(&f)
                    .then_some([f, y])
            })
            .collect();
        (pts.len() >= 2).then_some(("Eigenmode Dispersion f(m,n)", pts))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        vec![
            ("Off-Resonance", atlas_scalar_color(0.08, high_contrast)),
            ("Modal Skirt", atlas_scalar_color(0.48, high_contrast)),
            ("Resonant Ridge", atlas_scalar_color(0.94, high_contrast)),
        ]
    }
}

#[inline]
fn classify_chladni_cell(
    spatial_axes: bool,
    freq: f64,
    total_gain: f64,
    (best_df, best_coup): (f64, f64),
    (nearest_f_df, nearest_f_coup): (f64, f64),
) -> AtlasCell {
    let regime = if spatial_axes {
        if nearest_f_coup < 0.16 {
            ChladniModalRegime::NodalDeadZone
        } else if nearest_f_df <= 0.015 && nearest_f_coup >= 0.60 {
            ChladniModalRegime::LockedStandingMode
        } else if nearest_f_df <= 0.08 && nearest_f_coup >= 0.35 {
            ChladniModalRegime::ModalSkirt
        } else {
            ChladniModalRegime::OffResonanceDecay
        }
    } else if best_df <= (0.04 * freq).max(0.25) {
        if best_coup < 0.15 {
            ChladniModalRegime::NodalDeadZone
        } else {
            ChladniModalRegime::LockedStandingMode
        }
    } else if total_gain >= 1.5 {
        ChladniModalRegime::ModalSkirt
    } else {
        ChladniModalRegime::OffResonanceDecay
    };
    AtlasCell {
        regime: regime as u8,
        primary: total_gain,
        secondary: if spatial_axes {
            nearest_f_coup
        } else {
            best_df
        },
    }
}
