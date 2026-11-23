//! Downsampled `32x32` 2D FFT soliton viability cartographer (`LeniaCartographer` /
//! `LeniaSpectralCartographer`) wired into [`crate::framework::SimulationFramework`]'s
//! `FrameworkAtlasHost` for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
use super::LeniaSpeciesPreset;
use super::LeniaTool;
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer, atlas_regime_color,
};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::biology::lenia::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, LeniaProbeOutcome, LeniaProbeSpec,
    LeniaSpecies, MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MAX_LENIA_RULE_PARAMS, MIN_KERNEL_RADIUS,
    PROBE_CELLS, PROBE_STEPS, evaluate_lenia_probe_32_preseeded, fft_2d_forward_32,
    fft_2d_inverse_32, precompute_probe_kernel_hat_32, seed_probe_species_32,
};
use num_complex::Complex;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Supported `(x_key, y_key)` parameter axis pairs for [`LeniaCartographer`].
pub const LENIA_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("mu", "sigma"), ("radius", "mu"), ("mu", "radius")];

/// Integrated torus mass threshold below which a soliton is considered washed out on teleport.
pub const WASHOUT_MASS_THRESHOLD: f64 = 0.25;

/// Analytical slope $\mu / \sigma = \sqrt{2 \ln 2}$ along the Gaussian $G(0; \mu, \sigma) = 0$ locus.
#[inline]
#[must_use]
pub fn gaussian_vacuum_boundary_slope() -> f64 {
    (2.0_f64 * std::f64::consts::LN_2).sqrt()
}

/// Discrete qualitative dynamical regimes classified by [`LeniaCartographer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LeniaRegime {
    /// Initial soliton starves and decays to vacuum ($A \to 0$).
    Extinction = 0,
    /// Self-sustaining localized glider, shield, or dividing multi-lobe soliton.
    StableSoliton = 1,
    /// Outward-expanding wave or colony saturating the domain while $G(0) \le 0$.
    ExpandingWave = 2,
    /// Unstable empty vacuum ($G(0) > 0$) triggering spontaneous global explosion.
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

    /// Converts a domain [`LeniaProbeOutcome`] into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_probe(outcome: LeniaProbeOutcome) -> Self {
        match outcome {
            LeniaProbeOutcome::Extinction => Self::Extinction,
            LeniaProbeOutcome::LocalizedSoliton => Self::StableSoliton,
            LeniaProbeOutcome::OvergrownBlob => Self::ExpandingWave,
            LeniaProbeOutcome::VacuumExplosion => Self::VacuumExplosion,
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

fn lenia_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (resolved, label, min, max, step) = match key {
        "radius" => (
            "radius",
            "Kernel Radius (R)",
            MIN_KERNEL_RADIUS,
            MAX_KERNEL_RADIUS,
            0.5,
        ),
        "sigma" => ("sigma", "Growth Width (σ)", 0.002, 0.15, 0.001),
        _ => ("mu", "Growth Center (μ)", 0.01, 0.60, 0.005),
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved,
        label,
        ParameterConstraint { min, max, step },
    )
}

struct CartographerBuildInput<'a> {
    theory_params: &'a HashMap<String, ParameterConstraint>,
    axis_keys: (&'static str, &'static str),
    species: LeniaSpecies,
    radius: f64,
    ring_weights_slice: &'a [f64],
    growth: GrowthParams,
    dt: f64,
    custom_growth: Option<CompiledGrowthRule>,
    custom_kernel: Option<CompiledKernelRule>,
}

/// Zero-allocation `32x32` 2D FFT parameter-regime cartographer for Lenia solitons.
#[derive(Clone, Debug)]
pub struct LeniaCartographer {
    pub(super) axes: AtlasAxisPair,
    species: LeniaSpecies,
    seed_radius: f64,
    base_radius: f64,
    base_mu: f64,
    pub(super) base_sigma: f64,
    base_dt: f64,
    ring_weights: [f64; MAX_KERNEL_RINGS],
    ring_count: usize,
    pub(super) custom_growth: Option<CompiledGrowthRule>,
    pub(super) mu_slot: Option<usize>,
    sigma_slot: Option<usize>,
    custom_kernel: Option<CompiledKernelRule>,
    initial_state: [f64; PROBE_CELLS],
    initial_mass: f64,
    precomputed_kernel_hat: Option<[Complex<f64>; PROBE_CELLS]>,
    max_initial_u0: Option<f64>,
    non_negative_kernel: bool,
}

/// Type alias preserving both `LeniaCartographer` and `LeniaSpectralCartographer` names.
#[cfg(test)]
pub type LeniaSpectralCartographer = LeniaCartographer;

fn compute_max_initial_u0(
    initial_state: &[f64; PROBE_CELLS],
    k_hat: &[Complex<f64>; PROBE_CELLS],
) -> f64 {
    let mut freq = [Complex::new(0.0, 0.0); PROBE_CELLS];
    for (dst, &v) in freq.iter_mut().zip(initial_state.iter()) {
        *dst = Complex::new(v, 0.0);
    }
    fft_2d_forward_32(&mut freq);
    for (f, &h) in freq.iter_mut().zip(k_hat.iter()) {
        *f *= h;
    }
    fft_2d_inverse_32(&mut freq);
    freq.iter().fold(0.0_f64, |acc, c| acc.max(c.re))
}

fn pack_ring_weights(slice: &[f64]) -> ([f64; MAX_KERNEL_RINGS], usize) {
    let mut ring_weights = [0.0; MAX_KERNEL_RINGS];
    let ring_count = slice.len().clamp(1, MAX_KERNEL_RINGS);
    if slice.is_empty() {
        ring_weights[0] = 1.0;
    } else {
        ring_weights[..ring_count].copy_from_slice(&slice[..ring_count]);
    }
    (ring_weights, ring_count)
}

fn precompute_invariant_kernel_hat(
    inp: &CartographerBuildInput<'_>,
    ring_slice: &[f64],
) -> Option<[Complex<f64>; PROBE_CELLS]> {
    (inp.axis_keys.0 != "radius" && inp.axis_keys.1 != "radius").then(|| {
        let mut k_hat = [Complex::new(0.0, 0.0); PROBE_CELLS];
        let spec = LeniaProbeSpec {
            species: inp.species,
            seed_radius: inp.radius,
            kernel_radius: inp.radius,
            ring_weights: ring_slice,
            growth: inp.growth,
            dt: inp.dt,
            custom_growth: None,
            custom_kernel: inp.custom_kernel.as_ref().map(|r| (r, r.params())),
        };
        precompute_probe_kernel_hat_32(&spec, &mut k_hat);
        k_hat
    })
}

impl LeniaCartographer {
    /// Precomputes the `32x32` 2D FFT soliton seed and invariant kernel spectrum for `axis_keys`.
    #[cfg(test)]
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
        Self::from_input(CartographerBuildInput {
            theory_params,
            axis_keys,
            species: species.to_domain(),
            radius,
            ring_weights_slice,
            growth,
            dt,
            custom_growth: None,
            custom_kernel: None,
        })
    }

    /// Builds a snapshot [`LeniaCartographer`] from [`LeniaTool`], including active AST sandboxes.
    #[must_use]
    pub fn from_tool(tool: &LeniaTool, axis_keys: (&'static str, &'static str)) -> Self {
        let theory_params = tool.theory().theory_parameters();
        Self::from_input(CartographerBuildInput {
            theory_params: &theory_params,
            axis_keys,
            species: tool.species.to_domain(),
            radius: tool.radius,
            ring_weights_slice: &tool.ring_weights,
            growth: GrowthParams::new(tool.mu, tool.sigma),
            dt: tool.dt,
            custom_growth: tool.engine.custom_growth().cloned(),
            custom_kernel: tool.engine.custom_kernel().cloned(),
        })
    }

    fn from_input(inp: CartographerBuildInput<'_>) -> Self {
        let (ring_weights, ring_count) = pack_ring_weights(inp.ring_weights_slice);
        let (mu_slot, sigma_slot) = inp.custom_growth.as_ref().map_or((None, None), |rule| {
            let names = rule.compiled().param_names();
            (
                names.iter().position(|n| n == "mu"),
                names.iter().position(|n| n == "sigma"),
            )
        });
        let mut initial_state = [0.0; PROBE_CELLS];
        let initial_mass = seed_probe_species_32(inp.species, inp.radius, &mut initial_state);
        let precomputed_kernel_hat =
            precompute_invariant_kernel_hat(&inp, &ring_weights[..ring_count]);
        let max_initial_u0 = precomputed_kernel_hat
            .as_ref()
            .map(|kh| compute_max_initial_u0(&initial_state, kh));
        let non_negative_kernel =
            inp.custom_kernel.is_none() && ring_weights[..ring_count].iter().all(|&w| w >= 0.0);
        Self {
            axes: AtlasAxisPair {
                x: lenia_axis_spec(inp.theory_params, inp.axis_keys.0),
                y: lenia_axis_spec(inp.theory_params, inp.axis_keys.1),
            },
            species: inp.species,
            seed_radius: inp.radius,
            base_radius: inp.radius,
            base_mu: inp.growth.mu,
            base_sigma: inp.growth.sigma,
            base_dt: inp.dt,
            ring_weights,
            ring_count,
            custom_growth: inp.custom_growth,
            mu_slot,
            sigma_slot,
            custom_kernel: inp.custom_kernel,
            initial_state,
            initial_mass,
            precomputed_kernel_hat,
            max_initial_u0,
            non_negative_kernel,
        }
    }

    #[inline]
    fn resolve_coords(&self, p1: f64, p2: f64) -> (f64, f64, f64) {
        let mut r = self.base_radius;
        let mut mu = self.base_mu;
        let mut sigma = self.base_sigma;
        for (k, v) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match k {
                "radius" => r = v,
                "mu" => mu = v,
                "sigma" => sigma = v,
                _ => {}
            }
        }
        (r, mu, sigma)
    }

    #[inline]
    pub(super) fn stack_growth_params(
        &self,
        rule: &CompiledGrowthRule,
        mu: f64,
        sigma: f64,
    ) -> ([f64; MAX_LENIA_RULE_PARAMS], usize) {
        let src = rule.params();
        let mut buf = [0.0; MAX_LENIA_RULE_PARAMS];
        let len = src.len().min(MAX_LENIA_RULE_PARAMS);
        buf[..len].copy_from_slice(&src[..len]);
        if let Some(idx) = self.mu_slot.filter(|&i| i < len) {
            buf[idx] = mu;
        }
        if let Some(idx) = self.sigma_slot.filter(|&i| i < len) {
            buf[idx] = sigma;
        }
        (buf, len)
    }

    #[inline]
    fn fast_gaussian_bound(&self, mu: f64, sigma: f64) -> Option<AtlasCell> {
        if self.custom_growth.is_some() {
            return None;
        }
        if mu < gaussian_vacuum_boundary_slope() * sigma {
            return Some(AtlasCell {
                regime: LeniaRegime::VacuumExplosion as u8,
                primary: PROBE_CELLS as f64,
                secondary: 1.0,
            });
        }
        if let Some(max_u0) = self
            .max_initial_u0
            .filter(|&u| self.non_negative_kernel && u < mu)
        {
            let d = max_u0 - mu;
            let g_bound = 2.0 * (-0.5 * (d * d) / (sigma * sigma)).exp() - 1.0;
            let dt = if self.base_dt.is_finite() {
                self.base_dt.clamp(0.01, 0.5)
            } else {
                0.1
            };
            if g_bound < 0.0 && 1.0 + (PROBE_STEPS as f64) * dt * g_bound <= 0.0 {
                return Some(AtlasCell {
                    regime: LeniaRegime::Extinction as u8,
                    primary: 0.0,
                    secondary: 0.0,
                });
            }
        }
        None
    }
}

impl RegimeCartographer for LeniaCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (radius, mu, sigma) = self.resolve_coords(p1, p2);
        if !mu.is_finite()
            || !sigma.is_finite()
            || !radius.is_finite()
            || sigma <= 0.0
            || radius < MIN_KERNEL_RADIUS
        {
            return AtlasCell {
                regime: LeniaRegime::VacuumExplosion as u8,
                primary: 1.0,
                secondary: 1.0,
            };
        }
        if let Some(fast_cell) = self.fast_gaussian_bound(mu, sigma) {
            return fast_cell;
        }
        let mut g_buf = [0.0; MAX_LENIA_RULE_PARAMS];
        let custom_g = self.custom_growth.as_ref().map(|rule| {
            let (tmp, len) = self.stack_growth_params(rule, mu, sigma);
            g_buf[..len].copy_from_slice(&tmp[..len]);
            (rule, &g_buf[..len])
        });
        let custom_k = self
            .custom_kernel
            .as_ref()
            .map(|rule| (rule, rule.params()));

        let spec = LeniaProbeSpec {
            species: self.species,
            seed_radius: self.seed_radius,
            kernel_radius: radius,
            ring_weights: &self.ring_weights[..self.ring_count],
            growth: GrowthParams::new(mu, sigma),
            dt: self.base_dt,
            custom_growth: custom_g,
            custom_kernel: custom_k,
        };
        let summary = evaluate_lenia_probe_32_preseeded(
            &spec,
            &self.initial_state,
            self.initial_mass,
            self.precomputed_kernel_hat.as_ref(),
        );
        AtlasCell {
            regime: LeniaRegime::from_probe(summary.outcome) as u8,
            primary: summary.final_mass,
            secondary: summary.peak_potential,
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match LeniaRegime::from_u8(cell.regime) {
            LeniaRegime::Extinction => (cell.primary * 0.25).clamp(0.20, 0.55),
            LeniaRegime::StableSoliton => (cell.primary / 24.0).clamp(0.35, 1.0),
            LeniaRegime::ExpandingWave => (cell.primary / 160.0).clamp(0.30, 1.0),
            LeniaRegime::VacuumExplosion => 0.90,
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        LeniaRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "24-Step Probe Mass"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = LeniaRegime::from_u8(cell.regime);
        format!(
            "{} = {:.4}, {} = {:.4}\nRegime: {}\nProbe Mass: {:.2} | max(U): {:.3}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            reg.label(),
            cell.primary,
            cell.secondary
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        let pts = if let Some(rule) = &self.custom_growth {
            self.custom_bisected_contour(rule)?
        } else {
            self.gaussian_contour()?
        };
        Some(("G(0, μ, σ) = 0", pts))
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
}
