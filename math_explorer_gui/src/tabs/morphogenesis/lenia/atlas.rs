//! Downsampled `32x32` 2D FFT soliton viability atlas (`LeniaSpectralCartographer`)
//! wired into [`crate::async_sim::atlas::ParameterAtlasController`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::LeniaTool;
use crate::async_sim::atlas::{
    AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent, RegimeCartographer,
    atlas_regime_color,
};
use crate::framework::InteractiveTool;
use crate::framework::recovery::FiniteStateGuard;
use eframe::egui;
use math_explorer::biology::lenia::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, LeniaProbeOutcome, LeniaProbeSpec,
    LeniaSpecies, MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MAX_LENIA_RULE_PARAMS, MIN_KERNEL_RADIUS,
    PROBE_CELLS, evaluate_lenia_probe_32_preseeded, precompute_probe_kernel_hat_32,
    seed_probe_species_32,
};
use num_complex::Complex;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Supported `(x_key, y_key)` parameter axis pairs for [`LeniaSpectralCartographer`].
pub const LENIA_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[("mu", "sigma"), ("radius", "mu")];

/// Integrated torus mass threshold below which the organism is considered extinct/washed out.
pub const WASHOUT_MASS_THRESHOLD: f64 = 0.25;

/// Default serialized atlas axis keys for [`super::LeniaToolState`].
#[must_use]
pub fn default_lenia_atlas_axes() -> [String; 2] {
    ["mu".to_string(), "sigma".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_lenia_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "mu" && axes[1] == "sigma"
}

/// Analytical slope $\mu / \sigma = \sqrt{2 \ln 2}$ along the Gaussian $G(0; \mu, \sigma) = 0$ locus.
#[must_use]
pub fn gaussian_vacuum_boundary_slope() -> f64 {
    (2.0_f64 * std::f64::consts::LN_2).sqrt()
}

/// Discrete qualitative dynamical regimes classified by [`LeniaSpectralCartographer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LeniaRegime {
    /// Initial soliton starves and decays to vacuum ($A \to 0$).
    Extinction = 0,
    /// Self-sustaining localized glider, shield, or dividing multi-lobe soliton.
    LocalizedSoliton = 1,
    /// Unbounded halo expansion saturating the domain while $G(0) \le 0$.
    OvergrownBlob = 2,
    /// Unstable empty vacuum ($G(0) > 0$) triggering spontaneous global explosion.
    VacuumExplosion = 3,
}

impl LeniaRegime {
    /// Converts a raw `u8` cell regime index into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::Extinction,
            1 => Self::LocalizedSoliton,
            2 => Self::OvergrownBlob,
            _ => Self::VacuumExplosion,
        }
    }

    /// Converts a domain [`LeniaProbeOutcome`] into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_probe(outcome: LeniaProbeOutcome) -> Self {
        match outcome {
            LeniaProbeOutcome::Extinction => Self::Extinction,
            LeniaProbeOutcome::LocalizedSoliton => Self::LocalizedSoliton,
            LeniaProbeOutcome::OvergrownBlob => Self::OvergrownBlob,
            LeniaProbeOutcome::VacuumExplosion => Self::VacuumExplosion,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Extinction => "Extinction",
            Self::LocalizedSoliton => "Localized Soliton",
            Self::OvergrownBlob => "Overgrown Blob",
            Self::VacuumExplosion => "Vacuum Explosion",
        }
    }
}

fn lenia_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (resolved, label, min, max, step) = match key {
        "radius" => ("radius", "Kernel Radius (R)", MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS, 0.5),
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

/// Zero-allocation `32x32` 2D FFT parameter-regime cartographer for Lenia solitons.
#[derive(Clone, Debug)]
pub struct LeniaSpectralCartographer {
    axes: AtlasAxisPair,
    species: LeniaSpecies,
    seed_radius: f64,
    base_radius: f64,
    base_mu: f64,
    base_sigma: f64,
    base_dt: f64,
    ring_weights: [f64; MAX_KERNEL_RINGS],
    ring_count: usize,
    custom_growth: Option<CompiledGrowthRule>,
    mu_slot: Option<usize>,
    sigma_slot: Option<usize>,
    custom_kernel: Option<CompiledKernelRule>,
    initial_state: [f64; PROBE_CELLS],
    initial_mass: f64,
    precomputed_kernel_hat: Option<[Complex<f64>; PROBE_CELLS]>,
}

impl LeniaSpectralCartographer {
    /// Builds a snapshot [`LeniaSpectralCartographer`] from [`LeniaTool`]'s active genome and rules.
    #[must_use]
    pub fn from_tool(tool: &LeniaTool, axis_keys: (&'static str, &'static str)) -> Self {
        let theory_params = tool.theory().theory_parameters();
        let mut ring_weights = [0.0; MAX_KERNEL_RINGS];
        let ring_count = tool.ring_weights.len().clamp(1, MAX_KERNEL_RINGS);
        for (dst, &w) in ring_weights.iter_mut().zip(tool.ring_weights.iter().take(ring_count)) {
            *dst = w;
        }
        let custom_growth = tool.engine.custom_growth().cloned();
        let (mu_slot, sigma_slot) = if let Some(rule) = &custom_growth {
            let names = rule.compiled().param_names();
            (
                names.iter().position(|n| n == "mu"),
                names.iter().position(|n| n == "sigma"),
            )
        } else {
            (None, None)
        };
        let species = tool.species.to_domain();
        let custom_kernel = tool.engine.custom_kernel().cloned();
        let mut initial_state = [0.0; PROBE_CELLS];
        let initial_mass = seed_probe_species_32(species, tool.radius, &mut initial_state);
        let precomputed_kernel_hat = (axis_keys.0 != "radius" && axis_keys.1 != "radius").then(|| {
            let mut k_hat = [Complex::new(0.0, 0.0); PROBE_CELLS];
            let spec = LeniaProbeSpec {
                species,
                seed_radius: tool.radius,
                kernel_radius: tool.radius,
                ring_weights: &ring_weights[..ring_count],
                growth: GrowthParams::new(tool.mu, tool.sigma),
                dt: tool.dt,
                custom_growth: None,
                custom_kernel: custom_kernel.as_ref().map(|r| (r, r.params())),
            };
            precompute_probe_kernel_hat_32(&spec, &mut k_hat);
            k_hat
        });
        Self {
            axes: AtlasAxisPair {
                x: lenia_axis_spec(&theory_params, axis_keys.0),
                y: lenia_axis_spec(&theory_params, axis_keys.1),
            },
            species,
            seed_radius: tool.radius,
            base_radius: tool.radius,
            base_mu: tool.mu,
            base_sigma: tool.sigma,
            base_dt: tool.dt,
            ring_weights,
            ring_count,
            custom_growth,
            mu_slot,
            sigma_slot,
            custom_kernel,
            initial_state,
            initial_mass,
            precomputed_kernel_hat,
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
    fn stack_growth_params(
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

    fn gaussian_contour(&self) -> Option<Vec<[f64; 2]>> {
        let slope = gaussian_vacuum_boundary_slope();
        match (self.axes.x.key, self.axes.y.key) {
            ("mu", "sigma") => {
                let s_lo = self.axes.y.constraint.min.max(self.axes.x.constraint.min / slope);
                let s_hi = self.axes.y.constraint.max.min(self.axes.x.constraint.max / slope);
                if s_hi <= s_lo {
                    return None;
                }
                let pts = (0..=64)
                    .map(|idx| {
                        let t = (idx as f64) / 64.0;
                        let s = s_lo + t * (s_hi - s_lo);
                        [slope * s, s]
                    })
                    .collect();
                Some(pts)
            }
            ("radius", "mu") => {
                let mu_c = slope * self.base_sigma;
                (self.axes.y.constraint.min..=self.axes.y.constraint.max)
                    .contains(&mu_c)
                    .then(|| {
                        vec![
                            [self.axes.x.constraint.min, mu_c],
                            [self.axes.x.constraint.max, mu_c],
                        ]
                    })
            }
            _ => None,
        }
    }

    fn custom_bisected_contour(&self, rule: &CompiledGrowthRule) -> Option<Vec<[f64; 2]>> {
        self.mu_slot?;
        let eval_g0 = |mu: f64, sigma: f64| {
            let (buf, len) = self.stack_growth_params(rule, mu, sigma);
            rule.evaluate_with_params(0.0, &buf[..len])
        };
        let bisect_mu = |sigma: f64, lo_bound: f64, hi_bound: f64| -> Option<f64> {
            let (mut lo, mut hi) = (lo_bound, hi_bound);
            let (mut flo, fhi) = (eval_g0(lo, sigma), eval_g0(hi, sigma));
            if flo.signum() == fhi.signum() && flo.abs() > 1e-6 && fhi.abs() > 1e-6 {
                return None;
            }
            for _ in 0..36 {
                let mid = 0.5 * (lo + hi);
                let fm = eval_g0(mid, sigma);
                if fm == 0.0 || (flo > 0.0 && fm < 0.0) || (flo < 0.0 && fm > 0.0) {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            Some(0.5 * (lo + hi))
        };
        match (self.axes.x.key, self.axes.y.key) {
            ("mu", "sigma") => {
                let mut pts = Vec::with_capacity(65);
                for idx in 0..=64 {
                    let s = self.axes.y.constraint.min + (idx as f64 / 64.0) * self.axes.y.span();
                    if let Some(mu) = bisect_mu(s, self.axes.x.constraint.min, self.axes.x.constraint.max) {
                        pts.push([mu, s]);
                    }
                }
                (pts.len() >= 2).then_some(pts)
            }
            ("radius", "mu") => bisect_mu(
                self.base_sigma,
                self.axes.y.constraint.min,
                self.axes.y.constraint.max,
            )
            .map(|mu_c| vec![[self.axes.x.constraint.min, mu_c], [self.axes.x.constraint.max, mu_c]]),
            _ => None,
        }
    }
}

impl RegimeCartographer for LeniaSpectralCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (radius, mu, sigma) = self.resolve_coords(p1, p2);
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
            LeniaRegime::Extinction => (cell.primary * 0.25).clamp(0.05, 0.55),
            LeniaRegime::LocalizedSoliton => (cell.primary / 24.0).clamp(0.35, 1.0),
            LeniaRegime::OvergrownBlob => (cell.primary / 160.0).clamp(0.30, 1.0),
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
        Some(("Vacuum Stability G(0) = 0", pts))
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            LeniaRegime::Extinction,
            LeniaRegime::LocalizedSoliton,
            LeniaRegime::OvergrownBlob,
            LeniaRegime::VacuumExplosion,
        ]
        .into_iter()
        .map(|r| (r.label(), atlas_regime_color(r as u8, 4, 0.72, high_contrast)))
        .collect()
    }
}

impl LeniaTool {
    pub(crate) fn atlas_context_signature(&self) -> u64 {
        let (xk, yk) = self.atlas.axis_keys();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.species.hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        for (name, val) in [("radius", self.radius), ("mu", self.mu), ("sigma", self.sigma)] {
            if name != xk && name != yk {
                name.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
        for &w in &self.ring_weights {
            w.to_bits().hash(&mut h);
        }
        self.use_custom_kernel.hash(&mut h);
        self.growth_bar.state_fingerprint().hash(&mut h);
        self.kernel_bar.state_fingerprint().hash(&mut h);
        h.finish()
    }

    pub(crate) fn active_atlas_coords(&self) -> (f64, f64) {
        let (xk, yk) = self.atlas.axis_keys();
        let lookup = |k: &str| match k {
            "radius" => self.radius,
            "sigma" => self.sigma,
            _ => self.mu,
        };
        (lookup(xk), lookup(yk))
    }

    pub(crate) fn build_cartographer_arc(
        &self,
        axis_keys: (&'static str, &'static str),
    ) -> Arc<dyn RegimeCartographer> {
        Arc::new(LeniaSpectralCartographer::from_tool(self, axis_keys))
    }

    /// Applies an [`AtlasTeleportEvent`] to `LeniaTool`, auto-reseeding washed-out solitons
    /// (`total_mass < 0.25`) when teleporting into `Localized Soliton`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let lock_kernel = self.is_kernel_locked_by_challenge();
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "radius" if !lock_kernel => self.radius = val.clamp(MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS),
                "mu" => {
                    self.mu = val.clamp(0.01, 0.60);
                    self.growth_bar.set_param("mu", self.mu);
                }
                "sigma" => {
                    self.sigma = val.clamp(0.002, 0.15);
                    self.growth_bar.set_param("sigma", self.sigma);
                }
                _ => {}
            }
        }
        self.diagnostic = None;
        if !self.parameters_within_safe_bounds() {
            FiniteStateGuard::clamp_parameters(self);
        } else {
            self.sync_engine_parameters();
        }
        if ev.regime_label == LeniaRegime::LocalizedSoliton.label()
            && self.engine.telemetry().total_mass < WASHOUT_MASS_THRESHOLD
        {
            self.reseed_current_genome();
        } else {
            self.engine.refresh_telemetry();
            self.record_telemetry_sample();
            self.refresh_pixels_in_place();
            self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
        }
    }
}
