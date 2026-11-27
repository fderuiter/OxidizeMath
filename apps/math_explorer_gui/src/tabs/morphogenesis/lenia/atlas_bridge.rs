//! Analytical / bisected $G(0) = 0$ contour locus builders and `FrameworkAtlasHost` bridge
//! helpers for [`super::atlas::LeniaCartographer`] and [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    gaussian_vacuum_boundary_slope, LeniaCartographer, LeniaRegime, WASHOUT_MASS_THRESHOLD,
};
use super::LeniaTool;
use crate::async_sim::atlas::{AtlasTeleportEvent, RegimeCartographer};
use crate::framework::recovery::FiniteStateGuard;
use math_explorer::biology::lenia::{CompiledGrowthRule, MAX_KERNEL_RADIUS, MIN_KERNEL_RADIUS};
use std::hash::{Hash, Hasher};

/// Strict extinction mass floor used for automatic slider re-seeding.
const EXTINCTION_MASS_FLOOR: f64 = 1e-4;

impl LeniaCartographer {
    pub(super) fn gaussian_contour(&self) -> Option<Vec<[f64; 2]>> {
        let slope = gaussian_vacuum_boundary_slope();
        match (self.axes.x.key, self.axes.y.key) {
            ("mu", "sigma") => {
                let s_lo = self
                    .axes
                    .y
                    .constraint
                    .min
                    .max(self.axes.x.constraint.min / slope);
                let s_hi = self
                    .axes
                    .y
                    .constraint
                    .max
                    .min(self.axes.x.constraint.max / slope);
                if s_hi <= s_lo {
                    return None;
                }
                let pts = (0..=64)
                    .map(|idx| {
                        let s = s_lo + ((idx as f64) / 64.0) * (s_hi - s_lo);
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
            ("mu", "radius") => {
                let mu_c = slope * self.base_sigma;
                (self.axes.x.constraint.min..=self.axes.x.constraint.max)
                    .contains(&mu_c)
                    .then(|| {
                        vec![
                            [mu_c, self.axes.y.constraint.min],
                            [mu_c, self.axes.y.constraint.max],
                        ]
                    })
            }
            _ => None,
        }
    }

    fn bisect_custom_zero_mu(
        &self,
        rule: &CompiledGrowthRule,
        sigma: f64,
        lo_bound: f64,
        hi_bound: f64,
    ) -> Option<f64> {
        let eval_g0 = |mu: f64| {
            let (buf, len) = self.stack_growth_params(rule, mu, sigma);
            rule.evaluate_with_params(0.0, &buf[..len])
        };
        let (mut lo, mut hi) = (lo_bound, hi_bound);
        let (mut flo, fhi) = (eval_g0(lo), eval_g0(hi));
        if flo.signum() == fhi.signum() && flo.abs() > 1e-6 && fhi.abs() > 1e-6 {
            return None;
        }
        for _ in 0..36 {
            let mid = 0.5 * (lo + hi);
            let fm = eval_g0(mid);
            if fm == 0.0 || (flo > 0.0 && fm < 0.0) || (flo < 0.0 && fm > 0.0) {
                hi = mid;
            } else {
                lo = mid;
                flo = fm;
            }
        }
        Some(0.5 * (lo + hi))
    }

    pub(super) fn custom_bisected_contour(
        &self,
        rule: &CompiledGrowthRule,
    ) -> Option<Vec<[f64; 2]>> {
        self.mu_slot?;
        match (self.axes.x.key, self.axes.y.key) {
            ("mu", "sigma") => {
                let mut pts = Vec::with_capacity(65);
                for idx in 0..=64 {
                    let s = self.axes.y.constraint.min + (idx as f64 / 64.0) * self.axes.y.span();
                    if let Some(mu) = self.bisect_custom_zero_mu(
                        rule,
                        s,
                        self.axes.x.constraint.min,
                        self.axes.x.constraint.max,
                    ) {
                        pts.push([mu, s]);
                    }
                }
                (pts.len() >= 2).then_some(pts)
            }
            ("radius", "mu") => self
                .bisect_custom_zero_mu(
                    rule,
                    self.base_sigma,
                    self.axes.y.constraint.min,
                    self.axes.y.constraint.max,
                )
                .map(|mu_c| {
                    vec![
                        [self.axes.x.constraint.min, mu_c],
                        [self.axes.x.constraint.max, mu_c],
                    ]
                }),
            ("mu", "radius") => self
                .bisect_custom_zero_mu(
                    rule,
                    self.base_sigma,
                    self.axes.x.constraint.min,
                    self.axes.x.constraint.max,
                )
                .map(|mu_c| {
                    vec![
                        [mu_c, self.axes.y.constraint.min],
                        [mu_c, self.axes.y.constraint.max],
                    ]
                }),
            _ => None,
        }
    }
}

impl LeniaTool {
    /// Returns `true` when current `(mu, sigma, radius, dt)` lies inside a viable soliton or colony regime.
    #[must_use]
    pub fn is_current_niche_viable(&self) -> bool {
        let cart = LeniaCartographer::from_tool(self, ("mu", "sigma"));
        let cell = cart.evaluate_cell(self.mu, self.sigma);
        matches!(
            LeniaRegime::from_u8(cell.regime),
            LeniaRegime::StableSoliton | LeniaRegime::ExpandingWave
        )
    }

    /// Re-seeds the active species genome automatically when an extinct grid (`total_mass <= 1e-4`)
    /// enters a viable soliton niche.
    pub fn reseed_if_extinct_entering_viable(&mut self) -> bool {
        if self.engine.telemetry().total_mass <= EXTINCTION_MASS_FLOOR
            && self.is_current_niche_viable()
        {
            self.reseed_current_genome();
            return true;
        }
        false
    }

    #[cfg(test)]
    pub(crate) fn is_finite_and_extinct_for_test(&self) -> bool {
        let mass = self.engine.telemetry().total_mass;
        mass.is_finite() && mass <= EXTINCTION_MASS_FLOOR
    }

    pub(crate) fn lenia_atlas_context_sig(&self, axis_keys: (&'static str, &'static str)) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.species.hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.ring_weights.len().hash(&mut h);
        for &w in &self.ring_weights {
            w.to_bits().hash(&mut h);
        }
        for (name, val) in [
            ("radius", self.radius),
            ("mu", self.mu),
            ("sigma", self.sigma),
        ] {
            if name != axis_keys.0 && name != axis_keys.1 {
                name.hash(&mut h);
                val.to_bits().hash(&mut h);
            }
        }
        self.use_custom_kernel.hash(&mut h);
        self.growth_bar.state_fingerprint().hash(&mut h);
        self.kernel_bar.state_fingerprint().hash(&mut h);
        h.finish().max(1)
    }

    pub(crate) fn handle_lenia_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let lock_kernel = self.is_kernel_locked_by_challenge();
        for (key, val) in [(ev.x_key, ev.x_val), (ev.y_key, ev.y_val)] {
            match key {
                "radius" if !lock_kernel => {
                    self.radius = val.clamp(MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS);
                }
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
        let target_viable = ev.regime_label == LeniaRegime::StableSoliton.label()
            || ev.regime_label == LeniaRegime::ExpandingWave.label()
            || self.is_current_niche_viable();
        if self.engine.telemetry().total_mass < WASHOUT_MASS_THRESHOLD && target_viable {
            self.reseed_current_genome();
        } else {
            self.engine.refresh_telemetry();
            self.record_telemetry_sample();
            self.refresh_pixels_in_place();
            self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
        }
    }
}
