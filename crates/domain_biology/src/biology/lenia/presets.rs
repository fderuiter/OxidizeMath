//! Canonical continuous-lifeform species genomes (`Orbium`, `Geminium`, `Hydrogeminium`, `Scutium`)
//! and deterministic `OxidizeRng` seeding helpers.

use super::engine::LeniaEngine;
use super::kernel::{GrowthParams, KernelShells};
use math_commons::math_kernel::types::flatten_2d_index;
use oxidize_core::rng::OxidizeRng;
use rand::Rng;

const ORBIUM_DIM: usize = 20;

/// Canonical 20x20 `Orbium unicaudatus` glider cell matrix from Chan (2019).
#[rustfmt::skip]
const ORBIUM_CELLS: [f64; 400] = [
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.14, 0.10, 0.00, 0.00, 0.03, 0.03, 0.00, 0.00, 0.30, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.24, 0.30, 0.30, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.20, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.34, 0.44, 0.49, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.06, 0.13, 0.39, 0.50, 0.50, 0.37, 0.06, 0.00, 0.00, 0.00, 0.02, 0.16, 0.68, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.11, 0.17, 0.17, 0.33, 0.40, 0.38, 0.28, 0.14, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18, 0.42, 0.00, 0.00,
    0.00, 0.00, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.82, 0.00, 0.00,
    0.27, 0.00, 0.16, 0.12, 0.00, 0.00, 0.00, 0.25, 0.38, 0.44, 0.45, 0.34, 0.00, 0.00, 0.00, 0.00, 0.00, 0.22, 0.17, 0.00,
    0.00, 0.07, 0.20, 0.02, 0.00, 0.00, 0.00, 0.31, 0.48, 0.57, 0.60, 0.57, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.49, 0.00,
    0.00, 0.59, 0.19, 0.00, 0.00, 0.00, 0.00, 0.20, 0.57, 0.69, 0.76, 0.76, 0.49, 0.00, 0.00, 0.00, 0.00, 0.00, 0.36, 0.00,
    0.00, 0.58, 0.19, 0.00, 0.00, 0.00, 0.00, 0.00, 0.67, 0.83, 0.90, 0.92, 0.87, 0.12, 0.00, 0.00, 0.00, 0.00, 0.22, 0.07,
    0.00, 0.00, 0.46, 0.00, 0.00, 0.00, 0.00, 0.00, 0.70, 0.93, 1.00, 1.00, 1.00, 0.61, 0.00, 0.00, 0.00, 0.00, 0.18, 0.11,
    0.00, 0.00, 0.82, 0.00, 0.00, 0.00, 0.00, 0.00, 0.47, 1.00, 1.00, 0.98, 1.00, 0.96, 0.27, 0.00, 0.00, 0.00, 0.19, 0.10,
    0.00, 0.00, 0.46, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 1.00, 1.00, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.10, 0.21, 0.05,
    0.00, 0.00, 0.00, 0.40, 0.00, 0.00, 0.00, 0.00, 0.09, 0.80, 1.00, 0.82, 0.80, 0.85, 0.63, 0.31, 0.18, 0.19, 0.20, 0.01,
    0.00, 0.00, 0.00, 0.36, 0.10, 0.00, 0.00, 0.00, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.60, 0.39, 0.28, 0.24, 0.13, 0.00,
    0.00, 0.00, 0.00, 0.01, 0.30, 0.07, 0.00, 0.00, 0.08, 0.36, 0.64, 0.70, 0.64, 0.60, 0.51, 0.39, 0.29, 0.19, 0.04, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.10, 0.24, 0.14, 0.10, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.40, 0.31, 0.21, 0.08, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00,
];

/// Canonical Lenia soliton species presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LeniaSpecies {
    /// Smooth single-ring glider (`Orbium unicaudatus`).
    #[default]
    Orbium,
    /// Multi-ring twin-lobe dividing/oscillating organism (`Geminium`).
    Geminium,
    /// Three-ring wobbling swimmer (`Hydrogeminium natans`).
    Hydrogeminium,
    /// Resilient single-ring shield soliton (`Scutium gravidus`).
    Scutium,
}

/// Complete parametric genome for a [`LeniaSpecies`] organism.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LeniaGenome {
    /// Species identifier.
    pub species: LeniaSpecies,
    /// Concentric-ring kernel shell configuration.
    pub shells: KernelShells,
    /// Gaussian growth parameters $(\mu, \sigma)$.
    pub growth: GrowthParams,
    /// Canonical integration step size $\Delta t$.
    pub dt: f64,
}

impl LeniaSpecies {
    /// Ordered list of all canonical species presets.
    pub const ALL: [LeniaSpecies; 4] = [
        LeniaSpecies::Orbium,
        LeniaSpecies::Geminium,
        LeniaSpecies::Hydrogeminium,
        LeniaSpecies::Scutium,
    ];

    /// Human-readable species title.
    #[verified_engine::verified]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Orbium => "Orbium",
            Self::Geminium => "Geminium",
            Self::Hydrogeminium => "Hydrogeminium",
            Self::Scutium => "Scutium",
        }
    }

    /// Brief morphological description of the lifeform.
    #[verified_engine::verified]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Orbium => "Single-ring self-propelling glider soliton (Chan 2019).",
            Self::Geminium => "Three-ring twin-lobed dividing and undulating organism.",
            Self::Hydrogeminium => "Three-ring rotating and wobbling swimmer soliton.",
            Self::Scutium => "Dense single-ring shield organism with high perturbation resilience.",
        }
    }

    /// Returns the canonical [`LeniaGenome`] parameters for this species.
    #[verified_engine::verified]
    pub fn genome(&self) -> LeniaGenome {
        match self {
            Self::Orbium => LeniaGenome {
                species: *self,
                shells: KernelShells::new(13.0, &[1.0]),
                growth: GrowthParams::new(0.15, 0.015),
                dt: 0.1,
            },
            Self::Geminium => LeniaGenome {
                species: *self,
                shells: KernelShells::new(16.0, &[1.0, 0.55, 0.75]),
                growth: GrowthParams::new(0.24, 0.036),
                dt: 0.1,
            },
            Self::Hydrogeminium => LeniaGenome {
                species: *self,
                shells: KernelShells::new(15.0, &[0.5, 1.0, 0.667]),
                growth: GrowthParams::new(0.26, 0.036),
                dt: 0.1,
            },
            Self::Scutium => LeniaGenome {
                species: *self,
                shells: KernelShells::new(13.0, &[1.0]),
                growth: GrowthParams::new(0.26, 0.042),
                dt: 0.1,
            },
        }
    }
}

#[verified_engine::verified]
fn sample_orbium_bilinear(u: f64, v: f64) -> f64 {
    let max_coord = (ORBIUM_DIM - 1) as f64;
    if !(0.0..=max_coord).contains(&u) || !(0.0..=max_coord).contains(&v) {
        return 0.0;
    }
    let x0 = (u.floor() as usize).min(ORBIUM_DIM - 1);
    let y0 = (v.floor() as usize).min(ORBIUM_DIM - 1);
    let x1 = (x0 + 1).min(ORBIUM_DIM - 1);
    let y1 = (y0 + 1).min(ORBIUM_DIM - 1);
    let tx = u - (x0 as f64);
    let ty = v - (y0 as f64);
    let c00 = ORBIUM_CELLS[flatten_2d_index(x0, y0, ORBIUM_DIM)];
    let c10 = ORBIUM_CELLS[flatten_2d_index(x1, y0, ORBIUM_DIM)];
    let c01 = ORBIUM_CELLS[flatten_2d_index(x0, y1, ORBIUM_DIM)];
    let c11 = ORBIUM_CELLS[flatten_2d_index(x1, y1, ORBIUM_DIM)];
    let top = c00 * (1.0 - tx) + c10 * tx;
    let bot = c01 * (1.0 - tx) + c11 * tx;
    top * (1.0 - ty) + bot * ty
}

#[verified_engine::verified]
fn analytic_species_density(species: LeniaSpecies, nx: f64, ny: f64) -> f64 {
    let r = nx.hypot(ny);
    match species {
        LeniaSpecies::Orbium => {
            let u = nx * 13.0 + 9.5;
            let v = ny * 13.0 + 9.5;
            sample_orbium_bilinear(u, v)
        }
        LeniaSpecies::Scutium => {
            if r > 1.05 {
                return 0.0;
            }
            let ring = 0.92 * (-((r - 0.56) / 0.28).powi(2)).exp();
            let front_bias = 0.84 + 0.16 * (nx / (r + 0.05));
            (ring * front_bias).clamp(0.0, 1.0)
        }
        LeniaSpecies::Hydrogeminium => {
            if r > 1.45 {
                return 0.0;
            }
            let lobe_a = 0.98 * (-((nx + 0.34).hypot(ny - 0.12) / 0.58).powi(2)).exp();
            let lobe_b = 0.94 * (-((nx - 0.38).hypot(ny + 0.16) / 0.54).powi(2)).exp();
            let mantle = 0.72 * (-((r - 0.82) / 0.28).powi(2)).exp();
            (lobe_a.max(lobe_b) + 0.55 * mantle).clamp(0.0, 1.0)
        }
        LeniaSpecies::Geminium => {
            if r > 1.35 {
                return 0.0;
            }
            let twin_l = (-((nx + 0.36).hypot(ny - 0.06) / 0.42).powi(2)).exp();
            let twin_r = 0.95 * (-((nx - 0.36).hypot(ny + 0.08) / 0.42).powi(2)).exp();
            let bridge = 0.58 * (-(nx.hypot(ny - 0.16) / 0.36).powi(2)).exp();
            (twin_l + twin_r + bridge).clamp(0.0, 1.0)
        }
    }
}

/// Stamps a [`LeniaSpecies`] soliton centered at `(cx, cy)` using the engine's current radius
/// and deterministic `OxidizeRng` perturbation without clearing existing far-field organisms.
#[verified_engine::verified]
pub fn stamp_species_at(
    engine: &mut LeniaEngine,
    species: LeniaSpecies,
    cx: usize,
    cy: usize,
    perturbation: f64,
    rng: &mut OxidizeRng,
) {
    let radius = engine.shells().radius();
    let w = engine.width();
    let h = engine.height();
    let span = ((1.45 * radius).ceil() as isize).max(6);
    let noise_amp = if perturbation.is_finite() {
        perturbation.clamp(0.0, 0.35)
    } else {
        0.0
    };
    let grid = engine.state_mut();
    for dy in -span..=span {
        let y = (cy as isize + dy).rem_euclid(h as isize) as usize;
        for dx in -span..=span {
            let x = (cx as isize + dx).rem_euclid(w as isize) as usize;
            let base =
                analytic_species_density(species, (dx as f64) / radius, (dy as f64) / radius);
            if base > 1e-4 {
                let jitter = if noise_amp > 0.0 {
                    rng.gen_range(-noise_amp..=noise_amp)
                } else {
                    0.0
                };
                let idx = flatten_2d_index(x, y, w);
                let perturbed = base * (1.0 + jitter);
                grid[idx] = grid[idx].max(perturbed).clamp(0.0, 1.0);
            }
        }
    }
    engine.refresh_telemetry();
}

#[verified_engine::verified]
pub(crate) fn seed_engine_species(
    engine: &mut LeniaEngine,
    species: LeniaSpecies,
    perturbation: f64,
    rng: &mut OxidizeRng,
) {
    let genome = species.genome();
    engine.update_parameters(genome.shells, genome.growth, genome.dt);
    engine.clear();
    let cx = engine.width() / 2;
    let cy = engine.height() / 2;
    stamp_species_at(engine, species, cx, cy, perturbation, rng);
    engine.reset_clock_and_measure();
}
