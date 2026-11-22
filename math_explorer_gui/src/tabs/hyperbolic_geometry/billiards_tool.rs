#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{ExportCapabilities, InteractiveTool, ToolStateBounds};
use eframe::egui;
use math_explorer::num_complex::Complex64;
use math_explorer::pure_math::differential_geometry::hyperbolic::{
    HyperbolicBilliardConfig, HyperbolicBilliardSystem, HyperbolicModel, is_hyperbolic_schlafli,
};

const INITIAL_WARMUP_STEPS: usize = 120;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct HyperbolicBilliardsState {
    model: HyperbolicModel,
    p: usize,
    q: usize,
    initial_re: f64,
    initial_im: f64,
    initial_angle: f64,
    log10_epsilon: f64,
    step_ds: f64,
    steps_per_frame: usize,
}

/// Interactive twin-particle geodesic billiards inside regular $\{p, q\}$ hyperbolic polygons
/// with analytical circle-inversion reflections and finite-time Lyapunov divergence plotting.
pub struct HyperbolicBilliardsTool {
    pub model: HyperbolicModel,
    pub p: usize,
    pub q: usize,
    pub initial_re: f64,
    pub initial_im: f64,
    pub initial_angle: f64,
    pub log10_epsilon: f64,
    pub step_ds: f64,
    pub steps_per_frame: usize,
    pub is_playing: bool,
    pub system: HyperbolicBilliardSystem,
    pub revision: u64,
    pub last_canvas_id: Option<egui::Id>,
}

impl Default for HyperbolicBilliardsTool {
    fn default() -> Self {
        let cfg = HyperbolicBilliardConfig {
            p: 7,
            q: 3,
            initial_pos: Complex64::new(0.07, -0.04),
            initial_angle: 0.51,
            epsilon: 1e-6,
        };
        let mut system = HyperbolicBilliardSystem::new(cfg)
            .expect("default {7, 3} billiard configuration must succeed");
        for _ in 0..INITIAL_WARMUP_STEPS {
            system.step(0.035);
        }
        Self {
            model: HyperbolicModel::Poincare,
            p: 7,
            q: 3,
            initial_re: 0.07,
            initial_im: -0.04,
            initial_angle: 0.51,
            log10_epsilon: -6.0,
            step_ds: 0.035,
            steps_per_frame: 2,
            is_playing: false,
            system,
            revision: 1,
            last_canvas_id: None,
        }
    }
}

impl HyperbolicBilliardsTool {
    /// Current twin-particle perturbation $\epsilon = 10^{\log_{10} \epsilon}$.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        10.0_f64.powf(self.log10_epsilon.clamp(-9.0, -2.0))
    }

    /// Rebuilds the underlying [`HyperbolicBilliardSystem`] and optionally runs warmup steps.
    pub fn rebuild_system(&mut self, with_warmup: bool) -> Result<(), String> {
        let cfg = HyperbolicBilliardConfig {
            p: self.p,
            q: self.q,
            initial_pos: Complex64::new(self.initial_re, self.initial_im),
            initial_angle: self.initial_angle,
            epsilon: self.epsilon(),
        };
        self.system.reconfigure(cfg).map_err(|e| e.to_string())?;
        if with_warmup {
            for _ in 0..INITIAL_WARMUP_STEPS {
                self.system.step(self.step_ds);
            }
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    /// Extracts synchronized Lyapunov separation and theoretical $K = -1$ reference series.
    #[must_use]
    pub fn lyapunov_plot_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let n = self.system.sample_count();
        let mut log_curve = Vec::with_capacity(n.max(2));
        let mut ref_curve = Vec::with_capacity(n.max(2));
        for idx in 0..n {
            if let Some((_, _, sample)) = self.system.sample_at(idx) {
                log_curve.push([sample.path_length, sample.log_ratio]);
                ref_curve.push([sample.path_length, sample.path_length.min(18.0)]);
            }
        }
        if log_curve.len() < 2 {
            log_curve.push([0.0, 0.0]);
            log_curve.push([self.step_ds, self.step_ds]);
            ref_curve.push([0.0, 0.0]);
            ref_curve.push([1.0, 1.0]);
        }
        vec![
            ("Log Separation ln(d / eps)".to_string(), log_curve),
            ("Anosov K=-1 Slope (lambda = 1)".to_string(), ref_curve),
        ]
    }
}

impl InteractiveTool for HyperbolicBilliardsTool {
    fn name(&self) -> &'static str {
        "Hyperbolic Geodesic Billiards"
    }

    fn domain(&self) -> &'static str {
        "hyperbolic_geometry"
    }

    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.last_canvas_id
    }

    fn state_bounds(&self) -> ToolStateBounds {
        ToolStateBounds::DEFAULT
    }

    fn has_finite_parameters(&self) -> bool {
        is_hyperbolic_schlafli(self.p, self.q)
            && self.initial_re.is_finite()
            && self.initial_im.is_finite()
            && Complex64::new(self.initial_re, self.initial_im).norm() < 0.95
            && self.initial_angle.is_finite()
            && (-12.0..=12.0).contains(&self.initial_angle)
            && self.log10_epsilon.is_finite()
            && (-9.0..=-2.0).contains(&self.log10_epsilon)
            && self.step_ds.is_finite()
            && (0.005..=0.25).contains(&self.step_ds)
            && (1..=16).contains(&self.steps_per_frame)
    }

    fn clamp_parameters(&mut self) -> bool {
        if !is_hyperbolic_schlafli(self.p, self.q) {
            self.p = 7;
            self.q = 3;
        }
        if !self.initial_re.is_finite() {
            self.initial_re = 0.07;
        }
        if !self.initial_im.is_finite() {
            self.initial_im = -0.04;
        }
        let z = Complex64::new(self.initial_re, self.initial_im);
        let max_r = 0.85 * self.system.inradius_poincare().max(0.1);
        if z.norm() > max_r {
            let scaled = z * (max_r / z.norm());
            self.initial_re = scaled.re;
            self.initial_im = scaled.im;
        }
        if !self.initial_angle.is_finite() {
            self.initial_angle = 0.51;
        } else {
            self.initial_angle = self
                .initial_angle
                .clamp(-std::f64::consts::TAU, std::f64::consts::TAU);
        }
        if !self.log10_epsilon.is_finite() {
            self.log10_epsilon = -6.0;
        } else {
            self.log10_epsilon = self.log10_epsilon.clamp(-9.0, -2.0);
        }
        if !self.step_ds.is_finite() {
            self.step_ds = 0.035;
        } else {
            self.step_ds = self.step_ds.clamp(0.005, 0.2);
        }
        self.steps_per_frame = self.steps_per_frame.clamp(1, 12);
        self.is_playing = false;
        let _ = self.rebuild_system(true);
        true
    }

    fn save_state(&self) -> Option<String> {
        let state = HyperbolicBilliardsState {
            model: self.model,
            p: self.p,
            q: self.q,
            initial_re: self.initial_re,
            initial_im: self.initial_im,
            initial_angle: self.initial_angle,
            log10_epsilon: self.log10_epsilon,
            step_ds: self.step_ds,
            steps_per_frame: self.steps_per_frame,
        };
        serde_json::to_string(&state).ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: HyperbolicBilliardsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid HyperbolicBilliardsTool state: {e}"))?;
        if !is_hyperbolic_schlafli(s.p, s.q) {
            return Err(format!(
                "Schlafli symbol {{{}, {}}} is not hyperbolic",
                s.p, s.q
            ));
        }
        if !s.initial_re.is_finite()
            || !s.initial_im.is_finite()
            || Complex64::new(s.initial_re, s.initial_im).norm() >= 0.9
            || !s.initial_angle.is_finite()
            || !(-12.0..=12.0).contains(&s.initial_angle)
            || !s.log10_epsilon.is_finite()
            || !(-9.0..=-2.0).contains(&s.log10_epsilon)
            || !s.step_ds.is_finite()
            || !(0.005..=0.25).contains(&s.step_ds)
            || !(1..=16).contains(&s.steps_per_frame)
        {
            return Err("HyperbolicBilliardsTool parameters out of valid bounds".into());
        }
        self.model = s.model;
        self.p = s.p;
        self.q = s.q;
        self.initial_re = s.initial_re;
        self.initial_im = s.initial_im;
        self.initial_angle = s.initial_angle;
        self.log10_epsilon = s.log10_epsilon;
        self.step_ds = s.step_ds;
        self.steps_per_frame = s.steps_per_frame;
        self.is_playing = false;
        self.rebuild_system(true)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.render_billiards_tool(ui);
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot()
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.lyapunov_plot_datasets()
    }

    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        egui_plot::export_primitives::datasets_to_primitives(&self.export_datasets())
    }

    fn step_simulation(&mut self) {
        self.system.step(self.step_ds);
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

impl scientific_metadata::theory::TheoryDescribable for HyperbolicBilliardsTool {
    fn theory_description(&self) -> String {
        let sample = self.system.latest_sample();
        format!(
            "Geodesic billiards in a regular {{{}, {}}} hyperbolic polygon ($K = -1$). Nearby geodesics satisfy the Jacobi equation $J'' + K J = 0$, separating exponentially as $d(s) \\approx \\epsilon e^s$ coupled with circle-inversion wall reflections ($\\ln(d/\\epsilon) = {:.3}$, FTLE $\\lambda = {:.3}$).",
            self.p, self.q, sample.log_ratio, sample.ftle
        )
    }

    fn phonetic_description(&self) -> String {
        "Hyperbolic geodesic billiards with circle inversion reflections and Lyapunov divergence".into()
    }

    fn theory_citation(&self) -> String {
        "Hadamard (1898), Les surfaces a courbures opposees et leurs lignes geodesiques".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "HyperbolicBilliardsTool",
        display_name: "Hyperbolic Geodesic Billiards",
        domain: "hyperbolic_geometry",
        tags: &["hyperbolic", "billiards", "geodesic", "lyapunov", "chaos"],
        build: || Box::new(HyperbolicBilliardsTool::default()),
    }
}

const PRESET_BILLIARDS_7_3: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hyperbolic Geodesic Billiards - {7,3} Chaotic Divergence",
    "domain": "hyperbolic_geometry",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "model": "Poincare",
    "p": 7,
    "q": 3,
    "initial_re": 0.07,
    "initial_im": -0.04,
    "initial_angle": 0.51,
    "log10_epsilon": -6.0,
    "step_ds": 0.035,
    "steps_per_frame": 2
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "hyperbolic_billiards_7_3",
        title: "Chaotic {7,3} Hyperbolic Billiards",
        description: "Twin-particle geodesic billiards inside a {7,3} hyperbolic heptagon tracking exponential Lyapunov divergence.",
        category: "Differential Geometry",
        icon: "🎱",
        domain: "hyperbolic_geometry",
        tool_name: Some("HyperbolicBilliardsTool"),
        preset_generator: || PRESET_BILLIARDS_7_3,
    }
}

// [cite:modular_polynomials_review]
