#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

mod heatmap;
mod presets;
mod ui;

use crate::framework::{ExportCapabilities, InteractiveTool};
use eframe::egui;
use heatmap::MergedTensorHeatmap;
use math_explorer::applied::lorahub::{
    LoraEnsemble, LoraStateDict, NelderMeadConfig, NelderMeadSimplex, SimplexStepOutcome,
};
use math_explorer::nalgebra::DMatrix;
use scientific_metadata::theory::TheoryDescribable;

pub(crate) const ALPHA_MIN: f64 = 0.0;
pub(crate) const ALPHA_MAX: f64 = 1.50;
pub(crate) const STEP_MIN: f64 = 0.10;
pub(crate) const STEP_MAX: f64 = 0.80;
pub(crate) const BOUND_MIN: f64 = 0.50;
pub(crate) const BOUND_MAX: f64 = 1.50;
pub(crate) const ADAPTER_COUNT: usize = 4;
pub(crate) const TENSOR_DIM: usize = 6;
pub(crate) const ADAPTER_NAMES: [&str; ADAPTER_COUNT] =
    ["Math LoRA", "Code LoRA", "Logic LoRA", "Style LoRA"];

/// Cross-task target presets for dynamic LoRA weight composition.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum LoraTaskPreset {
    /// Symbolic math + code synthesis task favoring adapters 1 & 2 with sparse pruning of 3 & 4.
    #[default]
    MathCodeSynthesis,
    /// Cross-lingual summarization blending adapter 1 positively and adapter 2 negatively.
    CrossLingualSummary,
    /// Multi-skill algorithmic reasoning activating three adapters under L1 budget.
    AlgorithmicReasoning,
}

impl LoraTaskPreset {
    /// All selectable task presets.
    pub const ALL: [Self; 3] = [
        Self::MathCodeSynthesis,
        Self::CrossLingualSummary,
        Self::AlgorithmicReasoning,
    ];

    /// Display label for combo box controls.
    pub const fn label(self) -> &'static str {
        match self {
            Self::MathCodeSynthesis => "Math + Code Synthesis",
            Self::CrossLingualSummary => "Cross-Lingual Summary",
            Self::AlgorithmicReasoning => "Algorithmic Reasoning",
        }
    }

    pub(crate) fn target_and_stiffness(self) -> ([f64; ADAPTER_COUNT], [f64; ADAPTER_COUNT]) {
        match self {
            Self::MathCodeSynthesis => ([0.82, 0.58, 0.06, -0.04], [3.6, 3.2, 0.25, 0.20]),
            Self::CrossLingualSummary => ([0.70, -0.62, 0.09, 0.05], [3.4, 3.0, 0.22, 0.18]),
            Self::AlgorithmicReasoning => ([0.65, 0.48, -0.44, 0.05], [3.2, 2.8, 2.2, 0.20]),
        }
    }
}

fn build_adapter_ensemble() -> LoraEnsemble {
    let mut modules = Vec::with_capacity(ADAPTER_COUNT);
    for idx in 0..ADAPTER_COUNT {
        let phase = (idx as f64 + 1.0) * 0.75;
        let mat = DMatrix::from_fn(TENSOR_DIM, TENSOR_DIM, |r, c| {
            let rf = r as f64 / (TENSOR_DIM as f64);
            let cf = c as f64 / (TENSOR_DIM as f64);
            ((rf * 3.1 + phase).sin() * (cf * 2.7 - 0.5 * phase).cos()) * 0.85
                + if r == c && idx % 2 == 0 { 0.35 } else { 0.0 }
        });
        let mut dict = LoraStateDict::new();
        dict.insert("q_proj_delta".to_string(), mat);
        modules.push(dict);
    }
    LoraEnsemble::new(modules)
}

/// Interactive LoraHub gradient-free Nelder-Mead simplex composition playground.
pub struct LoraHubSimplexTool {
    /// L1 regularization coefficient $\alpha$ controlling adapter weight sparsity.
    pub alpha: f64,
    /// Initial coordinate perturbation step for simplex vertex construction.
    pub initial_step: f64,
    /// Symmetric box bound $[-B, B]$ on scalar adapter weights.
    pub weight_bound: f64,
    /// Active target task composition preset.
    pub task_preset: LoraTaskPreset,
    /// Whether continuous Nelder-Mead stepping is active.
    pub is_playing: bool,
    pub(super) ensemble: LoraEnsemble,
    pub(super) simplex: NelderMeadSimplex,
    pub(super) centroid_history: Vec<[f64; 2]>,
    pub(super) loss_history: Vec<[f64; 2]>,
    pub(super) merged_matrix: DMatrix<f64>,
    pub(super) heatmap: MergedTensorHeatmap,
    pub(super) last_outcome: SimplexStepOutcome,
    last_tick: Option<crate::web_io::PlatformInstant>,
}

impl Default for LoraHubSimplexTool {
    fn default() -> Self {
        let mut tool = Self {
            alpha: 0.35,
            initial_step: 0.42,
            weight_bound: 1.50,
            task_preset: LoraTaskPreset::MathCodeSynthesis,
            is_playing: false,
            ensemble: build_adapter_ensemble(),
            simplex: NelderMeadSimplex::new(ADAPTER_COUNT),
            centroid_history: Vec::new(),
            loss_history: Vec::new(),
            merged_matrix: DMatrix::zeros(TENSOR_DIM, TENSOR_DIM),
            heatmap: MergedTensorHeatmap::new(),
            last_outcome: SimplexStepOutcome::Reflected,
            last_tick: None,
        };
        tool.reset_simplex();
        tool
    }
}

impl LoraHubSimplexTool {
    pub(crate) fn simplex_config(&self) -> NelderMeadConfig {
        NelderMeadConfig {
            initial_step: self.initial_step.clamp(STEP_MIN, STEP_MAX),
            weight_bound: self.weight_bound.clamp(BOUND_MIN, BOUND_MAX),
            tolerance: 1e-6,
            max_steps: 200,
            ..NelderMeadConfig::default()
        }
    }

    /// Evaluates the unregularized few-shot task loss $L_{\text{task}}(w)$ for `weights`.
    pub fn evaluate_task_loss_for_preset(preset: LoraTaskPreset, weights: &[f64]) -> f64 {
        let (target, stiff) = preset.target_and_stiffness();
        let mut loss = 0.08;
        for i in 0..ADAPTER_COUNT.min(weights.len()) {
            let d = weights[i] - target[i];
            loss += stiff[i] * d * d;
        }
        if weights.len() >= 2 {
            let d0 = weights[0] - target[0];
            let d1 = weights[1] - target[1];
            loss += 0.45 * d0 * d1;
        }
        loss
    }

    /// Evaluates total LoraHub objective $L_{\text{task}}(w) + \frac{\alpha}{N}\sum_i |w_i|$.
    pub fn evaluate_objective_at(&self, weights: &[f64]) -> f64 {
        let task = Self::evaluate_task_loss_for_preset(self.task_preset, weights);
        self.ensemble
            .evaluate_objective(weights, task, self.alpha.clamp(ALPHA_MIN, ALPHA_MAX))
    }

    /// Resets the Nelder-Mead simplex to its initial displaced configuration and refreshes merged LoRA weights.
    pub fn reset_simplex(&mut self) {
        let cfg = self.simplex_config();
        let preset = self.task_preset;
        let alpha = self.alpha.clamp(ALPHA_MIN, ALPHA_MAX);
        let init = [-0.45, -0.35, 0.40, -0.30];
        let mut obj = |w: &[f64]| {
            let t_loss = Self::evaluate_task_loss_for_preset(preset, w);
            self.ensemble.evaluate_objective(w, t_loss, alpha)
        };
        self.simplex.initialize_in_place(&init, &cfg, &mut obj);
        self.centroid_history.clear();
        self.loss_history.clear();
        self.record_history_point();
        self.refresh_merged_tensor();
        self.last_outcome = SimplexStepOutcome::Reflected;
    }

    /// Advances the Nelder-Mead simplex by a single zero-allocation transition step.
    pub fn step_simplex_once(&mut self) {
        let cfg = self.simplex_config();
        let preset = self.task_preset;
        let alpha = self.alpha.clamp(ALPHA_MIN, ALPHA_MAX);
        let mut task_fn = |w: &[f64]| Self::evaluate_task_loss_for_preset(preset, w);
        self.last_outcome =
            self.ensemble
                .step_simplex_with_l1(&mut self.simplex, &cfg, alpha, &mut task_fn);
        self.record_history_point();
        self.refresh_merged_tensor();
        if self.last_outcome == SimplexStepOutcome::Converged {
            self.is_playing = false;
        }
    }

    pub(super) fn tick_playback(&mut self, ctx: &egui::Context) {
        if !self.is_playing {
            self.last_tick = None;
            return;
        }
        let now = crate::web_io::PlatformInstant::now();
        let should_step = match self.last_tick {
            Some(prev) => prev.elapsed().as_secs_f64() >= 0.08,
            None => true,
        };
        if should_step {
            self.step_simplex_once();
            self.last_tick = Some(now);
        }
        ctx.request_repaint();
    }

    fn record_history_point(&mut self) {
        let c = self.simplex.centroid();
        if c.len() >= 2 && self.centroid_history.len() < 256 {
            self.centroid_history.push([c[0], c[1]]);
        }
        if self.loss_history.len() < 256 {
            self.loss_history
                .push([self.simplex.steps_taken() as f64, self.simplex.best_value()]);
        }
    }

    fn refresh_merged_tensor(&mut self) {
        if let Ok(combined) = self.ensemble.combine(self.simplex.best_weights()) {
            if let Some(mat) = combined.get("q_proj_delta") {
                self.merged_matrix = mat.clone();
                self.heatmap.mark_dirty();
            }
        }
    }

    /// Builds closed 2D simplex vertex loop and iso-contour polylines for plotting and export.
    pub fn build_plot_series(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut simplex_ring = Vec::with_capacity(ADAPTER_COUNT + 2);
        for rank in 0..=ADAPTER_COUNT {
            let v = self.simplex.vertex(rank);
            simplex_ring.push([v[0], v[1]]);
        }
        if let Some(&first) = simplex_ring.first() {
            simplex_ring.push(first);
        }

        let best = self.simplex.best_weights();
        let (target, _) = self.task_preset.target_and_stiffness();
        let base_obj = self.evaluate_objective_at(&[target[0], target[1], best[2], best[3]]);
        let mut level_curves = Vec::new();
        for (idx, delta) in [0.12_f64, 0.35, 0.75].iter().enumerate() {
            let level = base_obj + delta;
            let mut ring = Vec::with_capacity(49);
            for k in 0..=48 {
                let theta = (k as f64 / 48.0) * std::f64::consts::TAU;
                let (ct, st) = (theta.cos(), theta.sin());
                let (mut lo, mut hi) = (0.0_f64, 1.8_f64);
                for _ in 0..10 {
                    let mid = 0.5 * (lo + hi);
                    let probe = [target[0] + mid * ct, target[1] + mid * st, best[2], best[3]];
                    if self.evaluate_objective_at(&probe) < level {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let r = 0.5 * (lo + hi);
                ring.push([
                    (target[0] + r * ct).clamp(-1.5, 1.5),
                    (target[1] + r * st).clamp(-1.5, 1.5),
                ]);
            }
            level_curves.push((format!("Objective Contour {}", idx + 1), ring));
        }

        let mut out = vec![
            ("Simplex Polytope".to_string(), simplex_ring),
            (
                "Centroid Trajectory".to_string(),
                self.centroid_history.clone(),
            ),
        ];
        out.extend(level_curves);
        out
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LoraHubState {
    alpha: f64,
    initial_step: f64,
    weight_bound: f64,
    #[serde(default)]
    task_preset: LoraTaskPreset,
}

impl InteractiveTool for LoraHubSimplexTool {
    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "LoraHub Simplex Composer"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_plot_series()
    }

    fn step_simulation(&mut self) {
        self.step_simplex_once();
    }

    fn state_revision(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.alpha.to_bits().hash(&mut h);
        self.initial_step.to_bits().hash(&mut h);
        self.weight_bound.to_bits().hash(&mut h);
        self.task_preset.hash(&mut h);
        self.simplex.steps_taken().hash(&mut h);
        h.finish().max(1)
    }

    fn has_finite_parameters(&self) -> bool {
        self.alpha.is_finite()
            && (ALPHA_MIN..=ALPHA_MAX).contains(&self.alpha)
            && self.initial_step.is_finite()
            && (STEP_MIN..=STEP_MAX).contains(&self.initial_step)
            && self.weight_bound.is_finite()
            && (BOUND_MIN..=BOUND_MAX).contains(&self.weight_bound)
    }

    fn clamp_parameters(&mut self) -> bool {
        self.alpha = if self.alpha.is_finite() {
            self.alpha.clamp(ALPHA_MIN, ALPHA_MAX)
        } else {
            0.35
        };
        self.initial_step = if self.initial_step.is_finite() {
            self.initial_step.clamp(STEP_MIN, STEP_MAX)
        } else {
            0.42
        };
        self.weight_bound = if self.weight_bound.is_finite() {
            self.weight_bound.clamp(BOUND_MIN, BOUND_MAX)
        } else {
            1.50
        };
        true
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&LoraHubState {
            alpha: self.alpha,
            initial_step: self.initial_step,
            weight_bound: self.weight_bound,
            task_preset: self.task_preset,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let parsed: LoraHubState =
            serde_json::from_str(state).map_err(|e| format!("Invalid LoraHub state: {e}"))?;
        if !parsed.alpha.is_finite() || !(ALPHA_MIN..=ALPHA_MAX).contains(&parsed.alpha) {
            return Err("alpha out of bounds".into());
        }
        if !parsed.initial_step.is_finite() || !(STEP_MIN..=STEP_MAX).contains(&parsed.initial_step)
        {
            return Err("initial_step out of bounds".into());
        }
        if !parsed.weight_bound.is_finite()
            || !(BOUND_MIN..=BOUND_MAX).contains(&parsed.weight_bound)
        {
            return Err("weight_bound out of bounds".into());
        }
        self.alpha = parsed.alpha;
        self.initial_step = parsed.initial_step;
        self.weight_bound = parsed.weight_bound;
        self.task_preset = parsed.task_preset;
        self.is_playing = false;
        self.reset_simplex();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.tick_playback(ui.ctx());
        self.draw_side_panel(ui);
        self.draw_central_panel(ui);
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

// [cite:lorahub]

#[cfg(test)]
mod tests;
