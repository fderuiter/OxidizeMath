//! Observables, declarative plot callouts, and the locked-kernel `"lenia_orbium_tail_regeneration"`
//! inverse challenge for [`super::LeniaTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::presets::{LeniaSpeciesPreset, LeniaToolState, default_brush_radius};
use super::{LeniaTool, sandbox};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{CalloutSpec, ChallengeEvaluation, ChallengeMetadata};
use math_explorer::biology::lenia::{MAX_KERNEL_RADIUS, MAX_KERNEL_RINGS, MIN_KERNEL_RADIUS};

/// Canonical ID of the `Orbium` tail-regeneration inverse challenge.
pub const CHALLENGE_ORBIUM_TAIL_REGEN: &str = "lenia_orbium_tail_regeneration";

const LOCKED_ORBIUM_RADIUS: f64 = 13.0;
const LOCKED_ORBIUM_GRID: usize = 64;
const LOCKED_ORBIUM_SEED: u64 = 42;
const TARGET_HEALED_MASS: f64 = 72.0;
const MIN_HEALED_MASS: f64 = 35.0;
const MAX_HEALED_MASS: f64 = 120.0;
const MIN_GLIDE_SPEED: f64 = 0.50;
const MAX_GLIDE_SPEED: f64 = 15.0;
const MIN_STABILIZED_TIME: f64 = 4.0;

fn generate_orbium_regen_baseline() -> String {
    let baseline = LeniaToolState {
        radius: LOCKED_ORBIUM_RADIUS,
        mu: 0.21,
        sigma: 0.012,
        dt: 0.1,
        ring_weights: vec![1.0],
        grid_size: LOCKED_ORBIUM_GRID,
        species: LeniaSpeciesPreset::Orbium,
        rng_seed: LOCKED_ORBIUM_SEED,
        perturbation: 0.0,
        brush_radius: 4.0,
        brush_mode: super::LeniaBrushMode::Slice,
        playing: false,
        growth_expr: None,
        growth_params: std::collections::BTreeMap::new(),
        kernel_expr: None,
        kernel_params: std::collections::BTreeMap::new(),
    };
    serde_json::to_string(&baseline).unwrap_or_default()
}

fn clamp_orbium_regen_locked_state(raw_json: &str) -> Result<String, String> {
    crate::url_state::URL_JSON_BOUNDS.validate_json(raw_json)?;
    let mut parsed: LeniaToolState =
        serde_json::from_str(raw_json).map_err(|e| format!("Invalid Lenia challenge JSON: {e}"))?;
    parsed.radius = LOCKED_ORBIUM_RADIUS;
    parsed.ring_weights = vec![1.0];
    parsed.species = LeniaSpeciesPreset::Orbium;
    parsed.grid_size = LOCKED_ORBIUM_GRID;
    parsed.rng_seed = LOCKED_ORBIUM_SEED;
    parsed.perturbation = 0.0;
    parsed.kernel_expr = None;
    parsed.kernel_params.clear();
    parsed.mu = if parsed.mu.is_finite() {
        parsed.mu.clamp(0.01, 0.60)
    } else {
        0.21
    };
    parsed.sigma = if parsed.sigma.is_finite() {
        parsed.sigma.abs().clamp(0.002, 0.15)
    } else {
        0.012
    };
    parsed.dt = if parsed.dt.is_finite() {
        parsed.dt.clamp(0.02, 0.20)
    } else {
        0.1
    };
    parsed.validate()?;
    serde_json::to_string(&parsed).map_err(|e| format!("Challenge state encode error: {e}"))
}

impl LeniaTool {
    /// Returns `true` when an active inverse challenge locks `radius`, `ring_weights`, and `K(r)`.
    #[inline]
    #[must_use]
    pub fn is_kernel_locked_by_challenge(&self) -> bool {
        self.active_challenge == Some(CHALLENGE_ORBIUM_TAIL_REGEN)
    }

    pub(crate) fn apply_orbium_tail_cut(&mut self) {
        let w = self.engine.width();
        let h = self.engine.height();
        let tail_x = (w / 2 + w - 7) % w;
        let tail_y = (h / 2 + 1) % h;
        self.engine.paint_disc(tail_x, tail_y, 4.5, -0.95);
    }

    pub(crate) fn collect_observables(&self) -> Vec<(&'static str, f64)> {
        let tel = self.engine.telemetry();
        vec![
            ("total_mass", tel.total_mass),
            ("active_area", tel.active_area),
            ("centroid_speed", tel.centroid_speed),
            ("max_potential", self.engine.max_potential()),
            ("mu", self.mu),
            ("sigma", self.sigma),
            ("radius", self.radius),
        ]
    }

    pub(crate) fn sync_declarative_callouts(&mut self, specs: &[CalloutSpec]) {
        self.callouts.sync_specs(specs);
    }

    pub(crate) fn set_active_challenge(
        &mut self,
        challenge_id: Option<&str>,
    ) -> Result<(), String> {
        match challenge_id {
            None => {
                self.active_challenge = None;
                self.callouts.clear_auto();
                Ok(())
            }
            Some(id) if id.eq_ignore_ascii_case(CHALLENGE_ORBIUM_TAIL_REGEN) => {
                let was_active = self.active_challenge == Some(CHALLENGE_ORBIUM_TAIL_REGEN);
                self.active_challenge = Some(CHALLENGE_ORBIUM_TAIL_REGEN);
                self.radius = LOCKED_ORBIUM_RADIUS;
                self.ring_weights = vec![1.0];
                self.species = LeniaSpeciesPreset::Orbium;
                self.grid_size = LOCKED_ORBIUM_GRID;
                self.rng_seed = LOCKED_ORBIUM_SEED;
                self.perturbation = 0.0;
                self.use_custom_kernel = false;
                if !was_active {
                    self.reseed_current_genome();
                } else {
                    self.sync_engine_parameters();
                }
                self.callouts.set_auto_callout(
                    "orbium_regen_target",
                    [0.15, 1.0],
                    "Viable Orbium healing niche (μ ≈ 0.15)",
                    [16.0, -24.0],
                );
                Ok(())
            }
            Some(other) => Err(format!("Unsupported LeniaTool challenge '{other}'")),
        }
    }

    pub(crate) fn evaluate_active_challenge(&self) -> Option<ChallengeEvaluation> {
        if self.active_challenge != Some(CHALLENGE_ORBIUM_TAIL_REGEN) {
            return None;
        }
        let tel = self.engine.telemetry();
        let mass_err = (tel.total_mass - TARGET_HEALED_MASS).abs();
        let mass_ok = (MIN_HEALED_MASS..=MAX_HEALED_MASS).contains(&tel.total_mass);
        let speed_ok = (MIN_GLIDE_SPEED..=MAX_GLIDE_SPEED).contains(&tel.centroid_speed);
        let time_ok = tel.time >= MIN_STABILIZED_TIME;
        let vacuum_safe = self.engine.evaluate_growth(0.0) <= 0.0;

        let mass_score = (1.0 - (mass_err / 40.0)).clamp(0.0, 1.0) * 45.0;
        let speed_score = (tel.centroid_speed / 3.0).clamp(0.0, 1.0) * 35.0;
        let time_score = (tel.time / MIN_STABILIZED_TIME).clamp(0.0, 1.0) * 20.0;
        let passed = mass_ok && speed_ok && time_ok && vacuum_safe;
        let score = if passed {
            100.0
        } else if !vacuum_safe {
            5.0
        } else {
            (mass_score + speed_score + time_score).min(95.0)
        };
        let detail = format!(
            "Mass {:.1} (target 35..120) | Speed {:.2} (>= 0.50) | t = {:.1}/{:.1}",
            tel.total_mass, tel.centroid_speed, tel.time, MIN_STABILIZED_TIME
        );
        Some(ChallengeEvaluation::new(
            score,
            passed,
            "|Mass - M*|",
            mass_err,
            detail,
        ))
    }

    pub(crate) fn parameters_within_safe_bounds(&self) -> bool {
        self.radius.is_finite()
            && (MIN_KERNEL_RADIUS..=MAX_KERNEL_RADIUS).contains(&self.radius)
            && self.mu.is_finite()
            && (0.01..=0.60).contains(&self.mu)
            && self.sigma.is_finite()
            && (0.002..=0.15).contains(&self.sigma)
            && self.dt.is_finite()
            && (0.005..=0.50).contains(&self.dt)
            && !self.ring_weights.is_empty()
            && self.ring_weights.len() <= MAX_KERNEL_RINGS
            && self
                .ring_weights
                .iter()
                .all(|w| w.is_finite() && (0.0..=1.0).contains(w))
            && self.ring_weights.iter().copied().sum::<f64>() > 1e-6
    }

    pub(crate) fn check_and_latch_instability(&mut self) -> bool {
        if !self.parameters_within_safe_bounds() {
            self.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.engine.telemetry().time,
                self.engine.telemetry().step_count as usize,
                "parameters",
                "Unstable Lenia step size or growth parameters detected; simulation halted by FiniteStateGuard.",
            ));
            return false;
        }
        if !self.engine.state().iter().all(|v| v.is_finite()) {
            self.playing = false;
            self.diagnostic = Some(DiagnosticReport::new(
                self.engine.telemetry().time,
                self.engine.telemetry().step_count as usize,
                "state",
                "Non-finite activation detected in Lenia field; simulation halted by FiniteStateGuard.",
            ));
            return false;
        }
        true
    }

    fn sanitize_rings_and_field(&mut self) {
        if self.ring_weights.is_empty() {
            self.ring_weights.push(1.0);
        }
        self.ring_weights.truncate(MAX_KERNEL_RINGS);
        for w in &mut self.ring_weights {
            *w = if w.is_finite() {
                w.clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        if self.ring_weights.iter().copied().sum::<f64>() <= 1e-6 {
            self.ring_weights[0] = 1.0;
        }
        if !self.engine.state().iter().all(|v| v.is_finite()) {
            for cell in self.engine.state_mut() {
                *cell = if cell.is_finite() {
                    cell.clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
            self.engine.refresh_telemetry();
        }
    }
}

impl FiniteStateGuard for LeniaTool {
    fn is_finite_state(&self) -> bool {
        self.diagnostic.is_none()
            && self.parameters_within_safe_bounds()
            && self.engine.state().iter().all(|v| v.is_finite())
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.diagnostic.as_ref()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let base_dt = if self.dt.is_finite() && self.dt > 0.0 {
            self.dt * 0.5
        } else {
            0.05
        };
        self.dt = base_dt.clamp(0.005, 0.25);
        FiniteStateGuard::clamp_parameters(self);
    }

    fn clamp_parameters(&mut self) {
        self.radius = if self.radius.is_finite() {
            self.radius.clamp(MIN_KERNEL_RADIUS, MAX_KERNEL_RADIUS)
        } else {
            13.0
        };
        self.mu = if self.mu.is_finite() {
            self.mu.clamp(0.01, 0.60)
        } else {
            0.15
        };
        self.sigma = if self.sigma.is_finite() {
            self.sigma.abs().clamp(0.002, 0.15)
        } else {
            0.015
        };
        self.dt = if self.dt.is_finite() {
            self.dt.clamp(0.005, 0.50)
        } else {
            0.1
        };
        self.sanitize_rings_and_field();
        self.diagnostic = None;
        self.sync_engine_parameters();
        self.refresh_pixels_in_place();
    }

    fn reset_parameters(&mut self) {
        self.playing = false;
        self.perturbation = 0.0;
        self.brush_radius = default_brush_radius();
        self.use_custom_kernel = false;
        self.growth_bar
            .set_expression(sandbox::DEFAULT_GAUSSIAN_GROWTH_EXPR);
        if self.is_kernel_locked_by_challenge() {
            self.mu = 0.21;
            self.sigma = 0.012;
            self.reseed_current_genome();
        } else {
            self.apply_species_preset(LeniaSpeciesPreset::Orbium);
        }
    }
}

inventory::submit! {
    ChallengeMetadata {
        id: CHALLENGE_ORBIUM_TAIL_REGEN,
        title: "Orbium Tail Regeneration",
        description: "A surgical slice removed part of an Orbium glider's tail while its initial growth niche (μ = 0.21) is too high, starving the wounded soliton. With Kernel R = 13 locked, tune Growth G(u) so the glider regenerates its tail and resumes steady translation.",
        category: "Artificial Life",
        icon: "🩹",
        domain: "biology",
        tool_name: "LeniaTool",
        locked_inputs: "Species = Orbium (tail-cut), Kernel R = 13.0, Single Ring β = [1.0], Grid = 64×64",
        tunable_budget: "Growth μ ∈ [0.01, 0.60], Width σ ∈ [0.002, 0.15], or custom G(u) formula",
        target_summary: "Stabilize healed Mass ∈ [35, 120] and Centroid Speed ≥ 0.50 after t ≥ 4.0",
        baseline_state_generator: generate_orbium_regen_baseline,
        clamp_locked_state: clamp_orbium_regen_locked_state,
    }
}
