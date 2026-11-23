//! Interactive **Lenia Studio** (`LeniaTool`) inside `MorphogenesisTab`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod atlas;
mod canvas;
pub mod challenge;
pub mod presets;
pub mod sandbox;
mod ui;

#[cfg(test)]
mod atlas_sandbox_tests;
#[cfg(test)]
mod tests;

pub use atlas::LENIA_ATLAS_AXIS_PAIRS;
pub use presets::{LeniaBrushMode, LeniaSpeciesPreset, LeniaTheory, LeniaToolState};
#[cfg(test)]
pub use sandbox::compute_potential_distribution;

use crate::async_sim::atlas::ParameterAtlasController;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    CalloutManager, CalloutSpec, ChallengeEvaluation, ExportCapabilities, InteractiveTool,
    MathExpressionInputBar, ToolStateBounds,
};
use eframe::egui;
use eframe::egui::ColorImage;
use math_explorer::biology::lenia::{GrowthParams, KernelShells, LeniaEngine, LeniaTelemetry};
use math_explorer::rng::OxidizeRng;
use presets::default_brush_radius;
use sandbox::{build_growth_sandbox, build_kernel_sandbox};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

const MAX_TELEMETRY_HISTORY: usize = 240;
const COOPERATIVE_FRAME_BUDGET_SECS: f64 = 0.008;

/// Interactive FFT-accelerated 2D Lenia continuous cellular automata studio.
pub struct LeniaTool {
    pub(crate) engine: LeniaEngine,
    /// Kernel interaction radius `R` (`4.0..=32.0`).
    pub radius: f64,
    /// Growth niche center `mu`.
    pub mu: f64,
    /// Growth niche width `sigma`.
    pub sigma: f64,
    /// Integration step `dt`.
    pub dt: f64,
    /// Active concentric ring weights (`1..=4`).
    pub ring_weights: Vec<f64>,
    /// Periodic square grid dimension (`32..=256`).
    pub grid_size: usize,
    /// Active species template.
    pub species: LeniaSpeciesPreset,
    /// Seeded `OxidizeRng` seed value.
    pub rng_seed: u64,
    /// Initial and stamp perturbation amplitude.
    pub perturbation: f64,
    /// Brush disc radius in grid cells.
    pub brush_radius: f64,
    /// Active mouse brush action.
    pub brush_mode: LeniaBrushMode,
    /// Whether continuous simulation playback is running.
    pub playing: bool,
    /// Progressive 2D `(mu, sigma)` / `(radius, mu)` soliton viability atlas controller.
    pub atlas: ParameterAtlasController,
    /// 1D formula sandbox for custom growth mapping `G(u)`.
    pub growth_bar: MathExpressionInputBar,
    /// 1D formula sandbox for custom radial kernel `K(r)`.
    pub kernel_bar: MathExpressionInputBar,
    /// Whether `kernel_bar` overrides concentric bump shells.
    pub use_custom_kernel: bool,
    /// Declarative and user-placed inspector plot callout annotations.
    pub callouts: CalloutManager,
    pub(crate) active_challenge: Option<&'static str>,
    pub(crate) cursor: [usize; 2],
    pub(crate) rng: OxidizeRng,
    pub(crate) texture: Option<egui::TextureHandle>,
    pub(crate) pixels: Vec<egui::Color32>,
    pub(crate) texture_dirty: bool,
    pub(crate) high_contrast: bool,
    pub(crate) history: VecDeque<LeniaTelemetry>,
    pub(crate) last_step_ms: f64,
    pub(crate) diagnostic: Option<DiagnosticReport>,
    pub(crate) focused_canvas: Option<egui::Id>,
    pub(crate) user_mutation_nonce: u64,
    theory: LeniaTheory,
}

impl Default for LeniaTool {
    fn default() -> Self {
        let grid_size = 128;
        let species = LeniaSpeciesPreset::Orbium;
        let rng_seed = 42;
        let mut rng = OxidizeRng::new(rng_seed);
        let engine =
            LeniaEngine::from_species(grid_size, grid_size, species.to_domain(), 0.0, &mut rng);
        let genome = species.to_domain().genome();
        let mut tool = Self {
            radius: genome.shells.radius(),
            mu: genome.growth.mu,
            sigma: genome.growth.sigma,
            dt: genome.dt,
            ring_weights: genome.shells.ring_weights().to_vec(),
            grid_size,
            species,
            rng_seed,
            perturbation: 0.0,
            brush_radius: default_brush_radius(),
            brush_mode: LeniaBrushMode::Deposit,
            playing: false,
            atlas: ParameterAtlasController::new(LENIA_ATLAS_AXIS_PAIRS.to_vec()),
            growth_bar: build_growth_sandbox(genome.growth.mu, genome.growth.sigma),
            kernel_bar: build_kernel_sandbox(),
            use_custom_kernel: false,
            callouts: CalloutManager::new(),
            active_challenge: None,
            cursor: [grid_size / 2, grid_size / 2],
            rng,
            engine,
            texture: None,
            pixels: vec![egui::Color32::BLACK; grid_size * grid_size],
            texture_dirty: true,
            high_contrast: false,
            history: VecDeque::with_capacity(MAX_TELEMETRY_HISTORY),
            last_step_ms: 0.0,
            diagnostic: None,
            focused_canvas: None,
            user_mutation_nonce: 0,
            theory: LeniaTheory::default(),
        };
        tool.record_telemetry_sample();
        tool.refresh_pixels_in_place();
        tool
    }
}

impl LeniaTool {
    /// Loads a canonical [`LeniaSpeciesPreset`] genome and reseeds the periodic grid via `OxidizeRng`.
    pub fn apply_species_preset(&mut self, species: LeniaSpeciesPreset) {
        if self.is_kernel_locked_by_challenge() {
            return;
        }
        self.species = species;
        let genome = species.to_domain().genome();
        self.radius = genome.shells.radius();
        self.ring_weights = genome.shells.ring_weights().to_vec();
        self.mu = genome.growth.mu;
        self.sigma = genome.growth.sigma;
        self.dt = genome.dt;
        self.push_mu_sigma_to_growth_bar();
        self.diagnostic = None;
        self.reseed_current_genome();
    }

    /// Re-seeds the active species spatial pattern while preserving current custom kernel/growth sliders.
    pub fn reseed_current_genome(&mut self) {
        self.rng = OxidizeRng::new(self.rng_seed);
        if self.engine.width() != self.grid_size || self.engine.height() != self.grid_size {
            self.engine = LeniaEngine::new(
                self.grid_size,
                self.grid_size,
                KernelShells::new(self.radius, &self.ring_weights),
                GrowthParams::new(self.mu, self.sigma),
                self.dt,
            );
            self.cursor = [self.grid_size / 2, self.grid_size / 2];
        }
        self.engine
            .seed_species(self.species.to_domain(), self.perturbation, &mut self.rng);
        self.sync_engine_parameters();
        if self.is_kernel_locked_by_challenge() {
            self.apply_orbium_tail_cut();
        }
        self.history.clear();
        self.record_telemetry_sample();
        self.refresh_pixels_in_place();
        self.user_mutation_nonce = self.user_mutation_nonce.wrapping_add(1);
    }

    pub(crate) fn sync_engine_parameters(&mut self) {
        if !self.parameters_within_safe_bounds() {
            return;
        }
        self.push_mu_sigma_to_growth_bar();
        let shells = KernelShells::new(self.radius, &self.ring_weights);
        let growth = GrowthParams::new(self.mu, self.sigma);
        self.engine.update_parameters(shells, growth, self.dt);
        self.sync_sandboxes_to_engine();
    }

    /// Executes a single timed spectral Lenia step using [`crate::web_io::PlatformInstant`].
    pub fn step_once(&mut self) {
        if !self.check_and_latch_instability() {
            return;
        }
        self.sync_engine_parameters();
        let start = crate::web_io::PlatformInstant::now();
        self.engine.step();
        self.last_step_ms = start.elapsed().as_secs_f64() * 1000.0;
        if self.check_and_latch_instability() {
            self.record_telemetry_sample();
            self.refresh_pixels_in_place();
        }
    }

    pub(crate) fn step_frame_cooperative(&mut self) {
        let frame_start = crate::web_io::PlatformInstant::now();
        for _ in 0..2 {
            self.step_once();
            if !self.playing || frame_start.elapsed().as_secs_f64() >= COOPERATIVE_FRAME_BUDGET_SECS
            {
                break;
            }
        }
    }

    pub(crate) fn record_telemetry_sample(&mut self) {
        if self.history.len() >= MAX_TELEMETRY_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(self.engine.telemetry());
    }

    fn current_state_snapshot(&self) -> LeniaToolState {
        let (xk, yk) = self.atlas.axis_keys();
        let sb = self.serialize_sandbox_fields();
        LeniaToolState {
            radius: self.radius,
            mu: self.mu,
            sigma: self.sigma,
            dt: self.dt,
            ring_weights: self.ring_weights.clone(),
            grid_size: self.grid_size,
            species: self.species,
            rng_seed: self.rng_seed,
            perturbation: self.perturbation,
            brush_radius: self.brush_radius,
            brush_mode: self.brush_mode,
            playing: self.playing,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
            growth_expr: sb.growth_expr,
            growth_params: sb.growth_params,
            kernel_expr: sb.kernel_expr,
            kernel_params: sb.kernel_params,
        }
    }
}

impl InteractiveTool for LeniaTool {
    fn name(&self) -> &'static str {
        "Lenia Studio"
    }

    fn domain(&self) -> &'static str {
        "biology"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        &self.theory
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        Vec::new()
    }

    fn export_raster_image(&self) -> Option<ColorImage> {
        let w = self.engine.width();
        let h = self.engine.height();
        (w > 0 && h > 0 && self.pixels.len() == w * h).then(|| ColorImage {
            size: [w, h],
            pixels: self.pixels.clone(),
            ..Default::default()
        })
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.collect_inspector_datasets()
    }

    fn step_simulation(&mut self) {
        for _ in 0..2 {
            self.step_once();
        }
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_studio(ui);
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.focused_canvas
    }

    fn has_finite_parameters(&self) -> bool {
        FiniteStateGuard::is_finite_state(self)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        self.collect_observables()
    }

    fn sync_callouts(&mut self, callouts: &[CalloutSpec]) {
        self.sync_declarative_callouts(callouts);
    }

    fn apply_challenge(&mut self, challenge_id: Option<&str>) -> Result<(), String> {
        self.set_active_challenge(challenge_id)
    }

    fn active_challenge_id(&self) -> Option<&'static str> {
        self.active_challenge
    }

    fn evaluate_challenge(&self) -> Option<ChallengeEvaluation> {
        self.evaluate_active_challenge()
    }

    fn clamp_parameters(&mut self) -> bool {
        FiniteStateGuard::clamp_parameters(self);
        true
    }

    fn state_bounds(&self) -> ToolStateBounds {
        crate::url_state::URL_JSON_BOUNDS
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.radius.to_bits().hash(&mut h);
        self.mu.to_bits().hash(&mut h);
        self.sigma.to_bits().hash(&mut h);
        self.dt.to_bits().hash(&mut h);
        self.ring_weights.len().hash(&mut h);
        for &w in &self.ring_weights {
            w.to_bits().hash(&mut h);
        }
        self.grid_size.hash(&mut h);
        self.species.hash(&mut h);
        self.rng_seed.hash(&mut h);
        self.perturbation.to_bits().hash(&mut h);
        self.brush_radius.to_bits().hash(&mut h);
        self.brush_mode.hash(&mut h);
        self.playing.hash(&mut h);
        self.atlas.is_visible().hash(&mut h);
        self.atlas.selected_pair_index().hash(&mut h);
        self.use_custom_kernel.hash(&mut h);
        self.growth_bar.state_fingerprint().hash(&mut h);
        self.kernel_bar.state_fingerprint().hash(&mut h);
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&self.current_state_snapshot()).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: LeniaToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Lenia state: {e}"))?;
        parsed.validate()?;
        let locked = self.is_kernel_locked_by_challenge();
        self.radius = if locked { 13.0 } else { parsed.radius };
        self.mu = parsed.mu;
        self.sigma = parsed.sigma;
        self.dt = parsed.dt;
        self.ring_weights = if locked {
            vec![1.0]
        } else {
            parsed.ring_weights
        };
        self.grid_size = if locked { 64 } else { parsed.grid_size };
        self.species = if locked {
            LeniaSpeciesPreset::Orbium
        } else {
            parsed.species
        };
        self.rng_seed = if locked { 42 } else { parsed.rng_seed };
        self.perturbation = if locked { 0.0 } else { parsed.perturbation };
        self.brush_radius = parsed.brush_radius;
        self.brush_mode = parsed.brush_mode;
        self.playing = parsed.playing;
        self.atlas
            .set_axis_keys(&parsed.atlas_axes[0], &parsed.atlas_axes[1])?;
        self.atlas.set_visible(parsed.atlas_visible);
        self.restore_sandbox_fields(
            parsed.growth_expr.as_deref(),
            &parsed.growth_params,
            parsed.kernel_expr.as_deref(),
            &parsed.kernel_params,
        );
        self.diagnostic = None;
        self.reseed_current_genome();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LeniaTool",
        display_name: "Lenia Studio",
        domain: "biology",
        tags: &[
            "lenia",
            "cellular-automata",
            "soliton",
            "orbium",
            "geminium",
            "hydrogeminium",
            "scutium",
            "fft",
        ],
        build: || Box::new(LeniaTool::default()),
    }
}
