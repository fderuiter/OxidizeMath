//! Zero-allocation 32-ring radial convolution surrogate cartographer and vacuum-stability
//! contour (`G(0, μ, σ) = 0`) for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::LeniaSpeciesPreset;
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer, atlas_regime_color,
};
use eframe::egui;
use math_explorer::biology::lenia::{GrowthParams, KernelShells, MAX_KERNEL_RINGS};
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Radial discretisation ring count for the stack-only 1D axisymmetric Lenia surrogate.
pub const LENIA_SURROGATE_RINGS: usize = 32;

/// Supported `(x, y)` parameter axis pairs for [`LeniaCartographer`].
pub const LENIA_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("mu", "sigma"), ("mu", "radius")];

/// Analytical vacuum-stability slope $\sqrt{2 \ln 2}$ where $G(0; \mu, \sigma) = 0$.
pub const VACUUM_STABILITY_SLOPE: f64 = 1.177_410_022_515_474_7;

const RING_DU: f64 = 2.4 / (LENIA_SURROGATE_RINGS as f64);
const ANGULAR_SAMPLES: usize = 8;
const SURROGATE_STEPS: usize = 30;

/// Precomputed $\cos(\theta_m)$ samples for $\theta_m = \pi (m + 0.5) / 8$.
const COS_THETA: [f64; ANGULAR_SAMPLES] = [
    0.980_785_280_403_230_4,
    0.831_469_612_302_545_2,
    0.555_570_233_019_602_2,
    0.195_090_322_016_128_25,
    -0.195_090_322_016_128_25,
    -0.555_570_233_019_602_2,
    -0.831_469_612_302_545_2,
    -0.980_785_280_403_230_4,
];

/// Discrete dynamical regimes predicted by the 32-ring radial Lenia surrogate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LeniaRegime {
    /// Soliton mass decays to zero under sub-threshold growth ($G(U) < 0$).
    Extinction = 0,
    /// Self-sustaining localized soliton with bounded radial support and stable vacuum.
    StableSoliton = 1,
    /// Outward-expanding wave / colony growth while empty vacuum remains stable ($G(0) \le 0$).
    ExpandingWave = 2,
    /// Spontaneous vacuum activation ($G(0; \mu, \sigma) > 0$, i.e. $\mu < \sqrt{2 \ln 2}\,\sigma$).
    VacuumExplosion = 3,
}

impl LeniaRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::Extinction,
            1 => Self::StableSoliton,
            2 => Self::ExpandingWave,
            _ => Self::VacuumExplosion,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Extinction => "Extinction",
            Self::StableSoliton => "Stable Soliton Niche",
            Self::ExpandingWave => "Expanding Colony",
            Self::VacuumExplosion => "Vacuum Explosion",
        }
    }
}

#[inline]
fn ring_u(idx: usize) -> f64 {
    ((idx as f64) + 0.5) * RING_DU
}

fn build_initial_radial_profile(species: LeniaSpeciesPreset) -> [f64; LENIA_SURROGATE_RINGS] {
    let mut profile = [0.0; LENIA_SURROGATE_RINGS];
    for (idx, slot) in profile.iter_mut().enumerate() {
        let nr = ring_u(idx);
        *slot = match species {
            LeniaSpeciesPreset::Orbium if nr <= 0.75 => {
                (0.92 * (-((nr - 0.34) / 0.115).powi(2)).exp()).clamp(0.0, 1.0)
            }
            LeniaSpeciesPreset::Scutium if nr <= 1.05 => {
                (0.88 * (-((nr - 0.34) / 0.28).powi(2)).exp()).clamp(0.0, 1.0)
            }
            LeniaSpeciesPreset::Geminium if nr <= 1.25 => {
                let c = 0.76 * (-((nr - 0.24) / 0.30).powi(2)).exp();
                (c + 0.38 * (-((nr - 0.68) / 0.22).powi(2)).exp()).clamp(0.0, 1.0)
            }
            LeniaSpeciesPreset::Hydrogeminium if nr <= 1.30 => {
                let c = 0.80 * (-((nr - 0.26) / 0.32).powi(2)).exp();
                (c + 0.44 * (-((nr - 0.72) / 0.24).powi(2)).exp()).clamp(0.0, 1.0)
            }
            _ => 0.0,
        };
    }
    profile
}

type PairDistGrid = [[[f64; ANGULAR_SAMPLES]; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS];

fn precompute_pair_distances() -> PairDistGrid {
    let mut dists = [[[0.0; ANGULAR_SAMPLES]; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS];
    for (i, row) in dists.iter_mut().enumerate() {
        let ui = ring_u(i);
        for (j, cell) in row.iter_mut().enumerate() {
            let uj = ring_u(j);
            for (m, &c) in COS_THETA.iter().enumerate() {
                cell[m] = (ui * ui + uj * uj - 2.0 * ui * uj * c).max(0.0).sqrt();
            }
        }
    }
    dists
}

fn build_radial_kernel_matrix(
    shells: &KernelShells,
    dists: &PairDistGrid,
) -> [[f64; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS] {
    let mut mat = [[0.0; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS];
    let rad = shells.radius().max(4.0);
    let inv_r = 1.0 / rad;
    for i in 0..LENIA_SURROGATE_RINGS {
        let mut row_sum = 0.0;
        for j in 0..LENIA_SURROGATE_RINGS {
            let mut ang = 0.0;
            for &dist in &dists[i][j] {
                ang += shells.shell_unnormalized((dist * rad).round() * inv_r);
            }
            let w = ang * ring_u(j) * RING_DU;
            mat[i][j] = w;
            row_sum += w;
        }
        if row_sum > 1e-12 {
            let inv_sum = 1.0 / row_sum;
            for val in &mut mat[i] {
                *val *= inv_sum;
            }
        }
    }
    mat
}

#[inline]
fn step_radial_surrogate(
    mat: &[[f64; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS],
    init: &[f64; LENIA_SURROGATE_RINGS],
    growth: GrowthParams,
    dt: f64,
    radius: f64,
) -> (f64, f64) {
    let mut a = *init;
    let step_dt = dt.clamp(0.05, 0.50);
    for _ in 0..SURROGATE_STEPS {
        let mut next = [0.0; LENIA_SURROGATE_RINGS];
        for i in 0..LENIA_SURROGATE_RINGS {
            let mut u = 0.0;
            for j in 0..LENIA_SURROGATE_RINGS {
                u += mat[i][j] * a[j];
            }
            next[i] = (a[i] + step_dt * growth.evaluate(u)).clamp(0.0, 1.0);
        }
        a = next;
    }
    let mut norm_mass = 0.0;
    for (idx, &val) in a.iter().enumerate() {
        norm_mass += val * ring_u(idx) * RING_DU * std::f64::consts::TAU;
    }
    let outer = a[LENIA_SURROGATE_RINGS - 1]
        .max(a[LENIA_SURROGATE_RINGS - 2])
        .max(a[LENIA_SURROGATE_RINGS - 3]);
    (norm_mass * radius * radius, outer)
}

fn lenia_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (label, min, max, step) = match key {
        "mu" => ("Growth Center (μ)", 0.01, 0.60, 0.005),
        "sigma" => ("Growth Width (σ)", 0.002, 0.15, 0.001),
        _ => ("Kernel Radius (R)", 4.0, 32.0, 0.5),
    };
    let resolved_key = if matches!(key, "mu" | "sigma" | "radius") {
        key
    } else {
        "radius"
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved_key,
        label,
        ParameterConstraint { min, max, step },
    )
}

/// Zero-allocation 32-ring analytical/surrogate 2D cartographer for [`super::LeniaTool`].
#[derive(Clone)]
pub struct LeniaCartographer {
    axes: AtlasAxisPair,
    base_mu: f64,
    base_sigma: f64,
    base_radius: f64,
    base_dt: f64,
    ring_weights: [f64; MAX_KERNEL_RINGS],
    ring_count: usize,
    radius_varies: bool,
    init_profile: [f64; LENIA_SURROGATE_RINGS],
    fixed_matrix: [[f64; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS],
    pair_dists: Box<[[[f64; ANGULAR_SAMPLES]; LENIA_SURROGATE_RINGS]; LENIA_SURROGATE_RINGS]>,
}

impl LeniaCartographer {
    /// Precomputes the 32-ring radial kernel geometry for `axis_keys`.
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        species: LeniaSpeciesPreset,
        radius: f64,
        ring_weights_slice: &[f64],
        growth: GrowthParams,
        dt: f64,
    ) -> Self {
        let mut ring_weights = [0.0; MAX_KERNEL_RINGS];
        let ring_count = ring_weights_slice.len().clamp(1, MAX_KERNEL_RINGS);
        if ring_weights_slice.is_empty() {
            ring_weights[0] = 1.0;
        } else {
            ring_weights[..ring_count].copy_from_slice(&ring_weights_slice[..ring_count]);
        }
        let pair_dists = Box::new(precompute_pair_distances());
        let base_shells = KernelShells::new(radius, &ring_weights[..ring_count]);
        let fixed_matrix = build_radial_kernel_matrix(&base_shells, &pair_dists);
        let init_profile = build_initial_radial_profile(species);
        let radius_varies = axis_keys.0 == "radius" || axis_keys.1 == "radius";

        Self {
            axes: AtlasAxisPair {
                x: lenia_axis_spec(theory_params, axis_keys.0),
                y: lenia_axis_spec(theory_params, axis_keys.1),
            },
            base_mu: growth.mu,
            base_sigma: growth.sigma,
            base_radius: radius,
            base_dt: dt,
            ring_weights,
            ring_count,
            radius_varies,
            init_profile,
            fixed_matrix,
            pair_dists,
        }
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64) {
        let mut mu = self.base_mu;
        let mut sigma = self.base_sigma;
        let mut radius = self.base_radius;
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "mu" => mu = val,
                "sigma" => sigma = val,
                "radius" => radius = val,
                _ => {}
            }
        }
        (mu, sigma, radius)
    }
}

impl RegimeCartographer for LeniaCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (mu, sigma, radius) = self.resolve_params(p1, p2);
        if !mu.is_finite()
            || !sigma.is_finite()
            || !radius.is_finite()
            || sigma <= 0.0
            || radius < 4.0
        {
            return AtlasCell {
                regime: LeniaRegime::VacuumExplosion as u8,
                primary: 1.0,
                secondary: 1.0,
            };
        }
        let growth = GrowthParams::new(mu, sigma);
        let g_zero = growth.evaluate(0.0);
        if g_zero > 0.0 {
            return AtlasCell {
                regime: LeniaRegime::VacuumExplosion as u8,
                primary: g_zero,
                secondary: 1.0,
            };
        }

        let (mass, outer_edge) = if self.radius_varies {
            let shells = KernelShells::new(radius, &self.ring_weights[..self.ring_count]);
            let mat = build_radial_kernel_matrix(&shells, &self.pair_dists);
            step_radial_surrogate(&mat, &self.init_profile, growth, self.base_dt, radius)
        } else {
            step_radial_surrogate(
                &self.fixed_matrix,
                &self.init_profile,
                growth,
                self.base_dt,
                radius,
            )
        };

        if mass <= 0.25 {
            AtlasCell {
                regime: LeniaRegime::Extinction as u8,
                primary: mass,
                secondary: g_zero,
            }
        } else if outer_edge >= 0.08 {
            AtlasCell {
                regime: LeniaRegime::ExpandingWave as u8,
                primary: mass,
                secondary: outer_edge,
            }
        } else {
            AtlasCell {
                regime: LeniaRegime::StableSoliton as u8,
                primary: mass,
                secondary: outer_edge,
            }
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match LeniaRegime::from_u8(cell.regime) {
            LeniaRegime::Extinction => 0.25,
            LeniaRegime::StableSoliton => (cell.primary / 280.0).clamp(0.35, 1.0),
            LeniaRegime::ExpandingWave => (cell.primary / 900.0).clamp(0.3, 1.0),
            LeniaRegime::VacuumExplosion => cell.primary.clamp(0.3, 1.0),
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        LeniaRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Surrogate Soliton Mass"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = LeniaRegime::from_u8(cell.regime);
        if reg == LeniaRegime::VacuumExplosion {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nVacuum Growth G(0): {:+.4} (> 0 unstable)",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary
            )
        } else {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nSurrogate Mass: {:.2}",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary
            )
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            LeniaRegime::Extinction,
            LeniaRegime::StableSoliton,
            LeniaRegime::ExpandingWave,
            LeniaRegime::VacuumExplosion,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.75, high_contrast),
            )
        })
        .collect()
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.axes.x.key != "mu" || self.axes.y.key != "sigma" {
            return None;
        }
        let s_min = self.axes.y.constraint.min;
        let s_max = self.axes.y.constraint.max;
        let mu_min = self.axes.x.constraint.min;
        let mu_max = self.axes.x.constraint.max;
        let mut pts = Vec::with_capacity(32);
        for idx in 0..=32 {
            let sigma = s_min + (idx as f64 / 32.0) * (s_max - s_min);
            let mu = VACUUM_STABILITY_SLOPE * sigma;
            if mu >= mu_min && mu <= mu_max {
                pts.push([mu, sigma]);
            }
        }
        (pts.len() >= 2).then_some(("G(0, μ, σ) = 0", pts))
    }
}
